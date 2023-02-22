package utc_test

import (
	"testing"

	"github.com/go-aie/paddle"
	"github.com/go-aie/utc"
	"github.com/google/go-cmp/cmp"
)

func TestPromptTokenizer_Encode(t *testing.T) {
	tests := []struct {
		inInputs     []utc.Input
		wantEncoding utc.Encoding
	}{
		{
			inInputs: []utc.Input{
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
			wantEncoding: utc.Encoding{
				InputIDs:     []int{1, 17964, 1566, 91, 17964, 955, 91, 2, 170, 5, 2, 2},
				SoftTokenIDs: []int{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				PositionIDs:  []int{0, 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 0},
				TokenTypeIDs: []int{0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0},
				AttentionMask: paddle.NewMatrix(12, 12, []float32{
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, -10000, -10000, -10000, 0, 0, 0, 0, 0,
					0, 0, 0, 0, -10000, -10000, -10000, 0, 0, 0, 0, 0,
					0, 0, 0, 0, -10000, -10000, -10000, 0, 0, 0, 0, 0,
					0, -10000, -10000, -10000, 0, 0, 0, 0, 0, 0, 0, 0,
					0, -10000, -10000, -10000, 0, 0, 0, 0, 0, 0, 0, 0,
					0, -10000, -10000, -10000, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				}),
				OMaskPositions: []int{1, 4},
				ClsPositions:   7,
			},
		},
	}
	for _, tt := range tests {
		tk, err := utc.NewPromptTokenizer("./utc-large/vocab.txt", true, 512)
		if err != nil {
			t.Fatalf("err: %v\n", err)
		}
		gotEncoding := tk.Encode(tt.inInputs)
		if !cmp.Equal(gotEncoding, tt.wantEncoding) {
			diff := cmp.Diff(gotEncoding, tt.wantEncoding)
			t.Errorf("Want - Got: %s", diff)
		}
	}
}
