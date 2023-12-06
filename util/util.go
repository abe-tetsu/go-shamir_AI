package util

import (
	"golang.org/x/xerrors"
	"math"
)

// Dot は2つのスライスのドット積（内積）を計算します。
func Dot(x, y []float64) (float64, error) {
	if len(x) != len(y) {
		return 0, xerrors.Errorf("both slices must have the same length, got %d and %d", len(x), len(y))
	}

	sum := 0.0
	for i := range x {
		sum += x[i] * y[i]
	}
	return sum, nil
}

func Relu(x []float64) []float64 {
	result := make([]float64, len(x))
	for i, val := range x {
		result[i] = math.Max(0, val)
	}
	return result
}

func Outer(x, y []float64) [][]float64 {
	rows := len(x)
	cols := len(y)
	result := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		result[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			result[i][j] = x[i] * y[j]
		}
	}
	return result
}
