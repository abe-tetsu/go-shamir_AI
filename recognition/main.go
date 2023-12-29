package main

import (
	"fmt"
	"github.com/abe-tetsu/shamir-ai/util"
	"github.com/petar/GoMNIST"
	"os"
)

func main() {
	fmt.Println("Start testing...")

	// MNISTデータをロード
	_, test, err := GoMNIST.Load("./data")
	if err != nil {
		fmt.Printf("failed to load mnist data: %v\n", err)
		os.Exit(1)
	}

	// 重みとバイアスをロード
	weights, err := util.LoadWeights("./ai-data/weights.gob")
	if err != nil {
		fmt.Printf("failed to load weights: %v\n", err)
		os.Exit(1)
	}

	biases, err := util.LoadBias("./ai-data/bias.gob")
	if err != nil {
		fmt.Printf("failed to load biases: %v\n", err)
		os.Exit(1)
	}

	// テストデータで精度を計算
	correctCount := 0
	for index := 0; index < len(test.Images); index++ {
		// 画像データの変換
		dataImage := util.TransformData(test.Images[index])

		// 順伝播の計算
		outputs := make([]float64, len(biases))
		for j := range outputs {
			for k := range weights {
				outputs[j] += dataImage[k] * weights[k][j]
			}
			outputs[j] += biases[j]
			outputs[j] = relu(outputs[j])
		}

		// 最も高い出力を持つラベルを選択
		maxIndex, maxValue := 0, outputs[0]
		for i, val := range outputs {
			if val > maxValue {
				maxIndex, maxValue = i, val
			}
		}

		// 正解数をカウント
		if maxIndex == int(test.Labels[index]) {
			correctCount++
		}
	}

	fmt.Println("テストケース数:", len(test.Images))
	fmt.Println("正解数　　　　:", correctCount)
	accuracy := float64(correctCount) / float64(len(test.Images))
	fmt.Printf("精度　　　　　: %.2f%%\n", accuracy*100)
}

func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}
