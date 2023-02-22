package utc

import (
	"math"

	"github.com/go-aie/paddle"
	"github.com/go-aie/xslices"
)

type Config struct {
	ModelPath, ParamsPath string
	VocabFile             string
	DoLowerCase           bool
	MaxSeqLength          int
	ForCN                 bool
	// The maximum number of predictors for concurrent inferences.
	// Defaults to the value of runtime.NumCPU.
	MaxConcurrency int
}

type Prediction struct {
	Text   string
	Scores map[string]float32
}

func (p Prediction) Best() (label string, score float32) {
	for l, s := range p.Scores {
		if s > score {
			score = s
			label = l
		}
	}
	return
}

type UTC struct {
	engine *paddle.Engine
	tk     *PromptTokenizer
}

func NewUTC(cfg *Config) *UTC {
	tk, err := NewPromptTokenizer(cfg.VocabFile, cfg.DoLowerCase, cfg.MaxSeqLength)
	if err != nil {
		panic(err)
	}

	return &UTC{
		engine: paddle.NewEngine(cfg.ModelPath, cfg.ParamsPath, cfg.MaxConcurrency),
		tk:     tk,
	}
}

func (u *UTC) Run(schema []string, texts []string) []Prediction {
	encodings := u.encode(schema, texts)

	inputs := u.getInputs(encodings)
	outputs := u.engine.Infer(inputs)

	result := outputs[0]
	m := paddle.NewMatrixFromTensor[float32](result)
	m.SetAllFunc(sigmoid[float32])

	var predictions []Prediction
	for i, row := range m.Rows() {
		scores := make(map[string]float32)
		for j := 0; j < len(row); j++ {
			label, score := schema[j], row[j]
			scores[label] = score
		}
		predictions = append(predictions, Prediction{
			Text:   texts[i],
			Scores: scores,
		})
	}
	return predictions
}

func (u *UTC) encode(schema []string, texts []string) []Encoding {
	var encodings []Encoding
	for _, text := range texts {
		inputs := BuildInputsWithPrompt(Example{
			TextA:   text,
			Choices: schema,
		})
		encodings = append(encodings, u.tk.Encode(inputs))
	}
	return encodings
}

func (u *UTC) getInputs(encodings []Encoding) []paddle.Tensor {
	var inputIDs [][]int64
	var tokenTypeIDs [][]int64
	var positionIDs [][]int64
	var attentionMaskMatrices []*paddle.Matrix[float32]
	var omaskPositions [][]int64
	var clsPositions []int64

	for _, e := range encodings {
		inputIDs = append(inputIDs, xslices.NumberToInt64(e.InputIDs))
		tokenTypeIDs = append(tokenTypeIDs, xslices.NumberToInt64(e.TokenTypeIDs))
		positionIDs = append(positionIDs, xslices.NumberToInt64(e.PositionIDs))
		attentionMaskMatrices = append(attentionMaskMatrices, e.AttentionMask)
		omaskPositions = append(omaskPositions, xslices.NumberToInt64(e.OMaskPositions))
		clsPositions = append(clsPositions, int64(e.ClsPositions))
	}

	// Do padding.
	inputIDs, _ = padRight(inputIDs, 0)
	tokenTypeIDs, _ = padRight(tokenTypeIDs, 0) // the token id of "[PAD]" is 0.
	positionIDs, _ = padRight(positionIDs, 0)
	paddedAttentionMaskMatrices := padMatrices(attentionMaskMatrices, -1e4)

	var attentionMask [][][][]float32
	for _, m := range paddedAttentionMaskMatrices {
		attentionMask = append(attentionMask, [][][]float32{m.Rows()}) // And one more dimension with a fixed-size 1.
	}

	return []paddle.Tensor{
		paddle.NewTensorFromTwoDimSlice(inputIDs),
		paddle.NewTensorFromTwoDimSlice(tokenTypeIDs),
		paddle.NewTensorFromTwoDimSlice(positionIDs),
		paddle.NewTensorFromFourDimSlice(attentionMask),
		paddle.NewTensorFromTwoDimSlice(omaskPositions),
		paddle.NewTensorFromOneDimSlice(clsPositions),
	}
}

// sigmoid transforms v to a new value between 0 and 1.
func sigmoid[T xslices.Number](v T) T {
	result := 1 / (1 + math.Exp(float64(-v)))
	return T(result)
}

// padRight pads the instances ss to the max sequence length in batch, and
// generate the corresponding mask, which is used to avoid attention on paddings.
func padRight[E xslices.Number](ss [][]E, padID E) (padded [][]E, mask [][]int) {
	maxLen := 0
	for _, inst := range ss {
		maxLen = xslices.Max(maxLen, len(inst))
	}

	for _, inst := range ss {
		paddedInst := inst
		var instMask []int
		for i := 0; i < len(inst); i++ {
			instMask = append(instMask, 1)
		}

		diffLen := maxLen - len(inst)
		if diffLen > 0 {
			paddedInst = make([]E, len(inst))
			copy(paddedInst, inst)

			for i := 0; i < diffLen; i++ {
				paddedInst = append(paddedInst, padID)
				instMask = append(instMask, 0)
			}
		}

		padded = append(padded, paddedInst)
		mask = append(mask, instMask)
	}

	return
}

func padMatrices[E xslices.Number](matrices []*paddle.Matrix[E], v E) (padded []*paddle.Matrix[E]) {
	var maxR, maxC int
	for _, m := range matrices {
		r, c := m.Dims()
		maxR = xslices.Max(maxR, r)
		maxC = xslices.Max(maxC, c)
	}

	for _, m := range matrices {
		r, c := m.Dims()
		pm := m.Pad(maxR-r, maxC-c, v)
		padded = append(padded, pm)
	}
	return
}
