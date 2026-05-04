package util

import (
	"math"
	"testing"
)

func TestEvalExpression(t *testing.T) {
	tests := []struct {
		expr    string
		want    float64
		wantErr bool
	}{
		{"2 + 2", 4, false},
		{"10 - 3", 7, false},
		{"3 * 4", 12, false},
		{"10 / 4", 2.5, false},
		{"10 % 3", 1, false},
		{"-5", -5, false},
		{"(2 + 3) * 4", 20, false},
		{"sqrt(4)", 2, false},
		{"pow(2, 10)", 1024, false},
		{"pi", math.Pi, false},
		{"e", math.E, false},
		{"0xff", 255, false},
		{"0b1010", 10, false},
		{"1 << 4", 16, false},
		{"255 & 0x0f", 15, false},
		{"min(3, 5)", 3, false},
		{"max(3, 5)", 5, false},
		{"abs(-7)", 7, false},
		{"floor(2.9)", 2, false},
		{"ceil(2.1)", 3, false},
		{"round(2.5)", 3, false},
		{"log2(8)", 3, false},
		{"2 / 0", 0, true},
		{"unknownfn(1)", 0, true},
		{"xyz", 0, true},
		{"", 0, true},
	}

	for _, tt := range tests {
		t.Run(tt.expr, func(t *testing.T) {
			got, err := evalExpression(tt.expr)
			if tt.wantErr {
				if err == nil {
					t.Errorf("evalExpression(%q) = %v, want error", tt.expr, got)
				}
				return
			}
			if err != nil {
				t.Errorf("evalExpression(%q) error: %v", tt.expr, err)
				return
			}
			if math.Abs(got-tt.want) > 1e-9 {
				t.Errorf("evalExpression(%q) = %v, want %v", tt.expr, got, tt.want)
			}
		})
	}
}

func TestFormatResult(t *testing.T) {
	tests := []struct {
		v    float64
		want string
	}{
		{4, "4"},
		{-5, "-5"},
		{2.5, "2.5"},
		{math.Inf(1), "+Inf"},
		{math.Inf(-1), "-Inf"},
		{math.NaN(), "NaN"},
		{1e15 + 1, "1.000000000000001e+15"}, // too large for int display
	}
	for _, tt := range tests {
		t.Run(tt.want, func(t *testing.T) {
			got := formatResult(tt.v)
			if got != tt.want {
				t.Errorf("formatResult(%v) = %q, want %q", tt.v, got, tt.want)
			}
		})
	}
}
