package util

import (
	"encoding/gob"
	"errors"
	"github.com/petar/GoMNIST"
	"image"
	"math"
	"os"
)

type MnistData struct {
	RawImage image.Image
	Label    int64
}

type MnistDataSet struct {
	DataSet []MnistData
	NCol    int
	NRow    int
}

func NewMnistDataSet(set *GoMNIST.Set) *MnistDataSet {
	dataSet := MnistDataSet{NCol: set.NCol, NRow: set.NRow}
	dataSet.DataSet = make([]MnistData, 0, set.Count())
	for i, rawData := range set.Images {
		data := newMnistDataFromGoMNISTData(rawData, int64(set.Labels[i]))
		dataSet.addData(data)
	}
	return &dataSet
}

func newMnistDataFromGoMNISTData(src GoMNIST.RawImage, label int64) MnistData {
	data := MnistData{src, label}
	return data
}

func (set *MnistDataSet) addData(data MnistData) {
	set.DataSet = append(set.DataSet, data)
}

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

func TransformData(data MnistData) []float64 {
	input := make([]float64, 784)
	for y := 0; y < 28; y++ {
		for x := 0; x < 28; x++ {
			pixel := data.RawImage.At(x, y)
			gray, _, _, _ := pixel.RGBA()
			input[y*28+x] = float64(gray) / 65535
		}
	}
	return input
}
