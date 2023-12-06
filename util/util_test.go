package util

import "testing"

func equal(x, y []float64) bool {
	if len(x) != len(y) {
		return false
	}
	for i, val := range x {
		if val != y[i] {
			return false
		}
	}
	return true
}

func Test_Dot(t *testing.T) {
	x := []float64{1, 2, 3}
	y := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
	}
	expected := []float64{9, 12, 15}
	actual := Dot(x, y)
	if !equal(actual, expected) {
		t.Errorf("Dot() = %v, want %v", actual, expected)
	}
}
