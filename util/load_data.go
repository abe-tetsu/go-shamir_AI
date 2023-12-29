package util

import (
	"encoding/gob"
	"errors"
	"image"
	"math"
	"os"
)

// LoadWeights は指定されたファイル名から重みの2次元配列を読み込み、NaNが含まれているかをチェックします。
func LoadWeights(fileName string) ([][]float64, error) {
	var weights [][]float64

	file, err := os.Open(fileName)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	if err := decoder.Decode(&weights); err != nil {
		return nil, err
	}

	// NaNをチェック
	for _, row := range weights {
		for _, weight := range row {
			if math.IsNaN(weight) {
				return nil, errors.New("weights contain NaN")
			}
		}
	}

	return weights, nil
}

// LoadBias は指定されたファイル名からバイアスの配列を読み込み、NaNが含まれているかをチェックします。
func LoadBias(fileName string) ([]float64, error) {
	var bias []float64

	file, err := os.Open(fileName)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	if err := decoder.Decode(&bias); err != nil {
		return nil, err
	}

	// NaNをチェック
	for _, b := range bias {
		if math.IsNaN(b) {
			return nil, errors.New("bias contain NaN")
		}
	}

	return bias, nil
}

func TransformData(data image.Image) []float64 {
	input := make([]float64, 784)
	for y := 0; y < 28; y++ {
		for x := 0; x < 28; x++ {
			pixel := data.At(x, y)
			gray, _, _, _ := pixel.RGBA()
			input[y*28+x] = float64(gray) / 6553500
		}
	}
	return input
}

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
