package main

import (
	"fmt"
	"github.com/abe-tetsu/shamir-ai/prediction"
	"github.com/abe-tetsu/shamir-ai/util"
	"github.com/petar/GoMNIST"
	"os"
)

func main() {
	// mnistデータをダウンロード
	err := util.DownloadMnist()
	if err != nil {
		fmt.Printf("failed to download mnist data: %w", err)
		os.Exit(2)
	}

	_, test, err := GoMNIST.Load("./data")
	if err != nil {
		fmt.Printf("failed to load mnist data: %w", err)
		os.Exit(2)
	}

	testDataSet := util.NewMnistDataSet(test)

	// 重みをロード
	weight, err := util.LoadWeights("./ai-data/weights.gob")
	if err != nil {
		fmt.Printf("failed to load weights: %w", err)
		os.Exit(2)
	}

	// バイアスをロード
	bias, err := util.LoadBias("./ai-data/bias.gob")
	if err != nil {
		fmt.Printf("failed to load bias: %w", err)
		os.Exit(2)
	}

	correctCount := 0
	for _, testData := range testDataSet.DataSet {
		// MNISTデータを適切な形式に変換
		input := util.TransformData(testData)

		result, err := prediction.Predict(input, weight, bias)
		if err != nil {
			fmt.Printf("failed to predict: %w", err)
			os.Exit(2)
		}

		// resultの中で最大の値を持つインデックスを取得
		maxIndex := 0
		for j := 0; j < len(result); j++ {
			if result[j] > result[maxIndex] {
				maxIndex = j
			}
		}

		// 予測結果と正解を比較
		if int64(maxIndex) == testData.Label {
			correctCount++
		}
	}

	// 正解率を出力
	fmt.Println("テストケース数: ", len(testDataSet.DataSet))
	fmt.Println("正解率　　　　: ", float64(correctCount)/float64(len(testDataSet.DataSet))*100, "%")
}
