package utc

import (
	"github.com/go-aie/paddle"
	aietokenizer "github.com/go-aie/tokenizer"
	"github.com/go-aie/xslices"
	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/model/wordpiece"
	"github.com/sugarme/tokenizer/normalizer"
	"github.com/sugarme/tokenizer/pretokenizer"
)

const (
	PadToken   = "[PAD]"
	OMaskToken = "[O-MASK]"
	ClsToken   = "[CLS]"
	SepToken   = "[SEP]"
)

type Encoding struct {
	SoftTokenIDs   []int
	InputIDs       []int
	PositionIDs    []int
	TokenTypeIDs   []int
	AttentionMask  *paddle.Matrix[float32]
	OMaskPositions []int
	ClsPositions   int
}

type PromptTokenizer struct {
	tk        *ErnieTokenizer
	maxLength int
}

func NewPromptTokenizer(vocabFile string, doLowerCase bool, maxSeqLength int) (*PromptTokenizer, error) {
	if maxSeqLength == 0 {
		maxSeqLength = 512
	}
	tk, err := NewErnieTokenizer(vocabFile, doLowerCase)
	if err != nil {
		return nil, err
	}
	return &PromptTokenizer{
		tk:        tk,
		maxLength: maxSeqLength,
	}, nil
}

func (t *PromptTokenizer) Encode(inputs []Input) Encoding {
	var softTokenIDs [][]int
	var positionIDs [][]int
	var tokenTypeIDs [][]int

	lastPositionID := 1 // Id 0 denotes special token "[CLS]".

	var inputIDs [][]int
	for _, input := range inputs {
		encs, err := t.tk.EncodeBatchTexts([]string{input.Text}, false)
		if err != nil {
			panic(err)
		}
		inputIDs = append(inputIDs, encs[0].Ids)
	}

	// TODO: Calculate max_lengths.

	for i, input := range inputs {
		partLen := len(inputIDs[i])
		softTokenIDs = append(softTokenIDs, xslices.Repeat([]int{0}, partLen))

		positionIDs = append(positionIDs, t.calcPositionIDsFromInput(input, inputIDs[i], &lastPositionID))

		tokenTypeIDs = append(tokenTypeIDs, xslices.Repeat([]int{input.TokenTypes}, partLen))
	}

	newInputIDs := t.tk.AddSpecialTokenIDs(xslices.Concat(inputIDs...), nil, t.tk.ClsTokenID(), t.tk.SepTokenID())
	newSoftTokenIDs := t.tk.AddSpecialTokenIDs(xslices.Concat(softTokenIDs...), nil, 0, 0)
	newPositionIDs := t.tk.AddSpecialTokenIDs(xslices.Concat(positionIDs...), nil, 0, 0)
	newTokenTypeIDs := t.tk.AddSpecialTokenIDs(xslices.Concat(tokenTypeIDs...), nil, 0, 0)

	attentionMask := t.calcOptionsAttentionMask(newInputIDs)

	omaskPositions := xslices.Indices(newInputIDs, t.tk.OMaskTokenID())
	// Not sure why this variable is named `clsPositions` while referring to the indices of "[SEP]" token.
	// Personally, I think the correct name is `sepPositions`.
	clsPositions := xslices.Indices(newInputIDs, t.tk.SepTokenID())[0]

	return Encoding{
		InputIDs:       newInputIDs,
		SoftTokenIDs:   newSoftTokenIDs,
		PositionIDs:    newPositionIDs,
		TokenTypeIDs:   newTokenTypeIDs,
		AttentionMask:  attentionMask,
		OMaskPositions: omaskPositions,
		ClsPositions:   clsPositions,
	}
}

func (t *PromptTokenizer) calcPositionIDsFromInput(input Input, inputIDs []int, lastPositionID *int) []int {
	if input.Positions >= 0 {
		*lastPositionID = input.Positions
	}

	var positionIDs []int

	var maxLen int
	for _, section := range t.splitByOMaskID(inputIDs) {
		positionIDs = append(positionIDs, xslices.Range(*lastPositionID, *lastPositionID+section.Len())...)
		maxLen = xslices.Max(maxLen, section.Len())
	}

	*lastPositionID += maxLen
	return positionIDs
}

func (t *PromptTokenizer) calcOptionsAttentionMask(inputIDs []int) *paddle.Matrix[float32] {
	length := len(inputIDs)
	attentionMask := paddle.NewMatrix[float32](length, length, nil).SetAll(1)

	omaskIndices := xslices.Indices(inputIDs, t.tk.OMaskTokenID())
	if len(omaskIndices) == 0 {
		// No need to calculate if there's no O-Mask token.
		return nil
	}
	lastOMaskIndex := omaskIndices[len(omaskIndices)-1]

	// Find the index of the first "[CLS]", which is greater than the last "[O-MASK]".
	clsIndex := length
	for _, idx := range xslices.Indices(inputIDs, t.tk.ClsTokenID()) {
		if idx > lastOMaskIndex {
			clsIndex = idx
			break
		}
	}

	// Find the index of the first "[SEP]", which is greater than the last "[O-MASK]".
	sepIndex := length
	for _, idx := range xslices.Indices(inputIDs, t.tk.SepTokenID()) {
		if idx > lastOMaskIndex {
			sepIndex = idx
			break
		}
	}

	optStart, optEnd := omaskIndices[0], xslices.Min(clsIndex, sepIndex)

	// Set the values of all elements, belonging to the "options" input, to 0.
	attentionMask.Set(optStart, optEnd, optStart, optEnd, 0)

	// Set the values of all elements, belonging to each "[O-MASK]" section, to 1.
	omaskIndices = append(omaskIndices, optEnd)
	for i := 0; i < len(omaskIndices)-1; i++ {
		idx, nextIdx := omaskIndices[i], omaskIndices[i+1]
		attentionMask.Set(idx, nextIdx, idx, nextIdx, 1)
	}

	// Invert all values to build the final attention mask.
	return attentionMask.SetAllFunc(func(v float32) float32 { return (v - 1) * 1e4 })
}

func (t *PromptTokenizer) splitByOMaskID(inputIDs []int) []omaskSection {
	var sections []omaskSection
	omaskID := t.tk.OMaskTokenID()

	var start int
	for i, inputID := range inputIDs {
		if inputID == omaskID {
			sections = append(sections, omaskSection{
				start: start,
				end:   i,
			})
			start = i
		}
	}
	sections = append(sections, omaskSection{
		start: start,
		end:   len(inputIDs),
	})

	return sections
}

type omaskSection struct {
	start, end int
}

func (s omaskSection) Len() int { return s.end - s.start }

type ErnieTokenizer struct {
	*aietokenizer.Tokenizer
}

func (t *ErnieTokenizer) PadTokenID() int {
	id, _ := t.Tokenizer.TokenToId(PadToken)
	return id
}

func (t *ErnieTokenizer) OMaskTokenID() int {
	id, _ := t.Tokenizer.TokenToId(OMaskToken)
	return id
}

func (t *ErnieTokenizer) ClsTokenID() int {
	id, _ := t.Tokenizer.TokenToId(ClsToken)
	return id
}

func (t *ErnieTokenizer) SepTokenID() int {
	id, _ := t.Tokenizer.TokenToId(SepToken)
	return id
}

func (t *ErnieTokenizer) AddSpecialTokenIDs(tokenIDsA, tokenIDsB []int, clsTokenID, sepTokenID int) []int {
	clsTokenIDs := []int{clsTokenID}
	sepTokenIDs := []int{sepTokenID}
	if len(tokenIDsB) == 0 {
		return xslices.Concat(clsTokenIDs, tokenIDsA, sepTokenIDs)
	}
	return xslices.Concat(clsTokenIDs, tokenIDsA, sepTokenIDs, tokenIDsB, sepTokenIDs)
}

func NewErnieTokenizer(vocabFile string, lowerCase bool) (*ErnieTokenizer, error) {
	m, err := wordpiece.NewWordPieceFromFile(vocabFile, "[UNK]")
	if err != nil {
		return nil, err
	}

	paddingStrategy := tokenizer.NewPaddingStrategy()
	paddingParams := tokenizer.PaddingParams{
		Strategy:  *paddingStrategy,
		Direction: tokenizer.Right, // padding right
	}
	tk := tokenizer.NewTokenizer(m)
	tk.AddSpecialTokens([]tokenizer.AddedToken{
		{Content: OMaskToken},
		{Content: ClsToken},
		{Content: SepToken},
	})

	tk.WithPadding(&paddingParams)
	tk.WithNormalizer(normalizer.NewBertNormalizer(true, lowerCase, true, false)) // Handle Chinese chars
	tk.WithPreTokenizer(pretokenizer.NewBertPreTokenizer())
	//tk.WithPostProcessor(processor.NewBertProcessing(processor.PostToken{}, processor.PostToken{}))

	return &ErnieTokenizer{
		Tokenizer: &aietokenizer.Tokenizer{Tokenizer: tk},
	}, nil
}
