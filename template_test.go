package utc_test

import (
	"testing"

	"github.com/go-aie/utc"
	"github.com/google/go-cmp/cmp"
)

func TestBuildInputsWithPrompt(t *testing.T) {
	tests := []struct {
		inExample  utc.Example
		wantResult []utc.Input
	}{
		{
			inExample: utc.Example{
				TextA:   "好的",
				Choices: []string{"肯定", "否定"},
			},
			wantResult: []utc.Input{
				{
					Text:       "[O-MASK]肯定[O-MASK]否定",
					Positions:  0,
					TokenTypes: 1,
					DoTruncate: false,
				},
				{
					Text:       "[SEP]",
					Positions:  0,
					TokenTypes: 0,
					DoTruncate: false,
				},
				{
					Text:       "好的",
					Positions:  -1,
					TokenTypes: 0,
					DoTruncate: true,
				},
				{
					Text:       "[SEP]",
					Positions:  -1,
					TokenTypes: 1,
					DoTruncate: false,
				},
				{
					Text:       "",
					Positions:  -1,
					TokenTypes: 1,
					DoTruncate: true,
				},
			},
		},
	}
	for _, tt := range tests {
		gotResult := utc.BuildInputsWithPrompt(tt.inExample)
		if !cmp.Equal(gotResult, tt.wantResult) {
			diff := cmp.Diff(gotResult, tt.wantResult)
			t.Errorf("Want - Got: %s", diff)
		}
	}
}
