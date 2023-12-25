package main

import (
	"fmt"
	"github.com/abe-tetsu/shamir-ai/util"
	"github.com/petar/GoMNIST"
	"image"
	"math/rand"
	"os"
)

func main() {
	fmt.Printf("\x1b[31m%s\x1b[0m", "start training............\n")

	// mnistデータをダウンロード
	err := util.DownloadMnist()
	if err != nil {
		fmt.Printf("failed to download mnist data: %w", err)
		os.Exit(2)
	}

	train, _, err := GoMNIST.Load("./data")
	if err != nil {
		fmt.Printf("failed to load mnist data: %w", err)
		os.Exit(2)
	}

	inputSize := 784 // 入力層のニューロン数
	outputSize := 10 // 出力層のニューロン数
	learningRate := 0.01
	epochs := 10

	// 重みとバイアスの初期化
	weights := make([][]float64, inputSize)
	for i := range weights {
		weights[i] = make([]float64, outputSize)
		for j := range weights[i] {
			weights[i][j] = rand.NormFloat64()
		}
	}
	biases := make([]float64, outputSize)
	for i := range biases {
		biases[i] = rand.Float64()
	}

	// 学習処理
	for epoch := 0; epoch < epochs; epoch++ {
		for index := 0; index < len(train.Images); index++ {
			// 画像の加工
			dataImage := TransformData(train.Images[index])

			// 順伝播の計算
			outputs := make([]float64, outputSize)
			for j := range outputs {
				for k := range weights {
					outputs[j] += dataImage[k] * weights[k][j]
				}
				outputs[j] += biases[j]
				outputs[j] = relu(outputs[j])
			}

			// 誤差の計算
			label := train.Labels[index]
			labels := make([]float64, outputSize)
			labels[label] = 1.0

			dz := make([]float64, outputSize)
			for j := range dz {
				dz[j] = outputs[j] - labels[j]
			}

			// outer積の計算
			dw := outer(dataImage, dz)

			// 重みとバイアスの更新
			for i := range weights {
				for j := range weights[i] {
					weights[i][j] -= learningRate * dw[i][j]
				}
			}
			for j := range biases {
				biases[j] -= learningRate * dz[j]
			}
		}
		fmt.Println("Epoch", epoch+1, "/", epochs)
	}

	// 重みとバイアスの保存
	err = util.SaveWeights(weights, "./ai-data/weights.gob")
	if err != nil {
		fmt.Printf("failed to save weights: %w", err)
		os.Exit(2)
	}

	err = util.SaveBias(biases, "./ai-data/bias.gob")
	if err != nil {
		fmt.Printf("failed to save bias: %w", err)
		os.Exit(2)
	}

	fmt.Printf("\x1b[32m%s\x1b[0m", "Training completed!\n")
}

func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func outer(x []float64, dz []float64) [][]float64 {
	dw := make([][]float64, len(x))
	for i := range x {
		dw[i] = make([]float64, len(dz))
		for j := range dz {
			dw[i][j] = x[i] * dz[j]
		}
	}
	return dw
}

func TransformData(data image.Image) []float64 {
	input := make([]float64, 784)
	for y := 0; y < 28; y++ {
		for x := 0; x < 28; x++ {
			pixel := data.At(x, y)
			gray, _, _, _ := pixel.RGBA()
			input[y*28+x] = float64(gray) / 65535
		}
	}
	return input
}
