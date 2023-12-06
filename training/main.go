package main

import (
	"fmt"
	"github.com/abe-tetsu/shamir-ai/network"
	"github.com/abe-tetsu/shamir-ai/util"
	"github.com/petar/GoMNIST"
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

	trainDataSet := util.NewMnistDataSet(train)

	// ニューラルネットワークの構造を定義
	nn := network.NewNeuralNetwork(784, 10, 5, 0.1)

	// ニューラルネットワークの学習
	_, err = nn.TrainNetWork(trainDataSet)
	if err != nil {
		fmt.Printf("failed to train network: %w", err)
		os.Exit(2)
	}

	fmt.Printf("\x1b[31m%s\x1b[0m", "..........finish training\n")
}
