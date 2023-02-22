package utc_test

import (
	"testing"

	"github.com/go-aie/utc"
	"github.com/google/go-cmp/cmp"
)

func TestUTC_Run(t *testing.T) {
	tests := []struct {
		inSchema        []string
		inText          []string
		wantPredictions []utc.Prediction
	}{
		{
			inSchema: []string{"肯定", "否定"},
			inText: []string{
				"好的",
				"不可以",
			},
			wantPredictions: []utc.Prediction{
				{
					Text: "好的",
					Scores: map[string]float32{
						"肯定": 0.72591889,
						"否定": 0.15622413,
					},
				},
				{
					Text: "不可以",
					Scores: map[string]float32{
						"肯定": 0.19716092,
						"否定": 0.68829225,
					},
				},
			},
		},
		{
			inSchema: []string{"肯定", "否定"},
			inText: []string{
				"很棒",
				"差评",
			},
			wantPredictions: []utc.Prediction{
				{
					Text: "很棒",
					Scores: map[string]float32{
						"肯定": 0.72918164,
						"否定": 0.21632886,
					},
				},
				{
					Text: "差评",
					Scores: map[string]float32{
						"肯定": 0.04616231,
						"否定": 0.92529543,
					},
				},
			},
		},
		{
			inSchema: []string{"肯定", "否定"},
			inText: []string{
				"awesome",
				"terrible",
			},
			wantPredictions: []utc.Prediction{
				{
					Text: "awesome",
					Scores: map[string]float32{
						"肯定": 0.68795687,
						"否定": 0.56465711,
					},
				},
				{
					Text: "terrible",
					Scores: map[string]float32{
						"肯定": 0.18670973,
						"否定": 0.86754951,
					},
				},
			},
		},
	}
	for _, tt := range tests {
		u := utc.NewUTC(&utc.Config{
			ModelPath:   "./utc-large/static/inference.pdmodel",
			ParamsPath:  "./utc-large/static/inference.pdiparams",
			VocabFile:   "./utc-large/vocab.txt",
			DoLowerCase: true,
		})
		gotPredictions := u.Run(tt.inSchema, tt.inText)
		if !cmp.Equal(gotPredictions, tt.wantPredictions) {
			diff := cmp.Diff(gotPredictions, tt.wantPredictions)
			t.Errorf("Want - Got: %s", diff)
		}
	}
}
