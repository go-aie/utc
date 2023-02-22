package utc

import "strings"

// DefaultMaxOptions is the default value for some template attributes.
const (
	DefaultMaxOptions = 10
)

type Example struct {
	TextA    string
	TextB    string
	Question string
	Choices  []string
}

type Prompt interface {
	BuildText(Example) string
	Positions() int
	TokenTypes() int
	DoTruncate() bool
}

type BasePrompt struct {
	Position  int
	TokenType int
	Truncate  bool
}

func (p BasePrompt) Positions() int   { return p.Position }
func (p BasePrompt) TokenTypes() int  { return p.TokenType }
func (p BasePrompt) DoTruncate() bool { return p.Truncate }

type ChoicesPrompt struct {
	BasePrompt

	AddOMask  bool
	AddPrompt bool
	Length    int
}

func (p ChoicesPrompt) BuildText(e Example) string {
	var labels []string
	for _, c := range e.Choices {
		if p.AddOMask {
			c = OMaskToken + c
		}
		labels = append(labels, c)
	}
	return strings.Join(labels, "")
}

type SepPrompt struct {
	BasePrompt

	Sep string
}

func (p SepPrompt) BuildText(e Example) string {
	return SepToken
}

type TextPrompt struct {
	BasePrompt

	Text string
}

func (p TextPrompt) BuildText(e Example) string {
	switch p.Text {
	case "text_a":
		return e.TextA
	case "text_b":
		return e.TextB
	}
	return ""
}

var (
	defaultPrompts = []Prompt{
		ChoicesPrompt{
			BasePrompt: BasePrompt{
				Position:  0,
				TokenType: 1,
				Truncate:  false,
			},
			AddOMask: true,
			Length:   DefaultMaxOptions,
		},
		SepPrompt{
			BasePrompt: BasePrompt{
				Position:  0,
				TokenType: 0,
				Truncate:  false,
			},
		},
		TextPrompt{
			BasePrompt: BasePrompt{
				Position:  -1,
				TokenType: 0, // Same as the previous one.
				Truncate:  true,
			},
			Text: "text_a",
		},
		SepPrompt{
			BasePrompt: BasePrompt{
				Position:  -1,
				TokenType: 1,
				Truncate:  false,
			},
		},
		TextPrompt{
			BasePrompt: BasePrompt{
				Position:  -1,
				TokenType: 1, // Same as the previous one.
				Truncate:  true,
			},
			Text: "text_b",
		},
	}
)

type Input struct {
	Text       string
	Positions  int
	TokenTypes int
	DoTruncate bool
}

func BuildInputsWithPrompt(e Example, ps ...Prompt) []Input {
	if len(ps) == 0 {
		ps = defaultPrompts
	}

	var inputs []Input
	for _, p := range ps {
		inputs = append(inputs, Input{
			Text:       p.BuildText(e),
			Positions:  p.Positions(),
			TokenTypes: p.TokenTypes(),
			DoTruncate: p.DoTruncate(),
		})
	}
	return inputs
}
