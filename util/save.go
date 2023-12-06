package util

import (
	"encoding/gob"
	"errors"
	"math"
	"os"
)

// containsNaNInMatrix は2次元配列内にNaNが含まれているかチェックします。
func containsNaNInMatrix(matrix [][]float64) bool {
	for _, row := range matrix {
		for _, value := range row {
			if math.IsNaN(value) {
				return true
			}
		}
	}
	return false
}

// containsNaNInSlice は1次元配列内にNaNが含まれているかチェックします。
func containsNaNInSlice(slice []float64) bool {
	for _, value := range slice {
		if math.IsNaN(value) {
			return true
		}
	}
	return false
}

// SaveWeights は重みをファイルに保存します。
// 重みの要素にNaNが含まれている場合はエラーを返します。
func SaveWeights(weights [][]float64, fileName string) error {
	if containsNaNInMatrix(weights) {
		return errors.New("weights contain NaN values")
	}

	file, err := os.Create(fileName)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	if err := encoder.Encode(weights); err != nil {
		return err
	}

	return nil
}

// SaveBias はバイアスをファイルに保存します。
// バイアスの要素にNaNが含まれている場合はエラーを返します。
func SaveBias(bias []float64, fileName string) error {
	if containsNaNInSlice(bias) {
		return errors.New("bias contains NaN values")
	}

	file, err := os.Create(fileName)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	if err := encoder.Encode(bias); err != nil {
		return err
	}

	return nil
}
