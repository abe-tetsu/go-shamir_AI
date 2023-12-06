package util

import (
	"io/ioutil"
	"net/http"
	"os"
)

const (
	TRAIN_URL      = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
	LABEL_URL      = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
	TEST_TRAIN_URL = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
	TEST_LABEL_URL = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"

	LOCAL_DATA_PATH        = "data"
	LOCAL_TRAIN_PATH       = LOCAL_DATA_PATH + "/train-images-idx3-ubyte.gz"
	LOCAL_LABELS_PATH      = LOCAL_DATA_PATH + "/train-labels-idx1-ubyte.gz"
	LOCAL_TEST_TRAIN_PATH  = LOCAL_DATA_PATH + "/t10k-images-idx3-ubyte.gz"
	LOCAL_TEST_LABELS_PATH = LOCAL_DATA_PATH + "/t10k-labels-idx1-ubyte.gz"
)

func DownloadMnist() error {
	urls := []string{TRAIN_URL, LABEL_URL, TEST_TRAIN_URL, TEST_LABEL_URL}
	localPaths := []string{LOCAL_TRAIN_PATH, LOCAL_LABELS_PATH, LOCAL_TEST_TRAIN_PATH, LOCAL_TEST_LABELS_PATH}
	if !FileExists(LOCAL_TRAIN_PATH) ||
		!FileExists(LOCAL_LABELS_PATH) ||
		!FileExists(LOCAL_TEST_TRAIN_PATH) ||
		!FileExists(LOCAL_TEST_LABELS_PATH) {
		if !FileExists(LOCAL_DATA_PATH) {
			if err := os.Mkdir(LOCAL_DATA_PATH, os.ModePerm); err != nil {
				return err
			}
		}

		for i := 0; i < len(urls); i++ {

			response, err := http.Get(urls[i])
			if err != nil {
				return err
			}

			body, err := ioutil.ReadAll(response.Body)
			if err != nil {
				return err
			}
			if err := ioutil.WriteFile(localPaths[i], body, os.ModePerm); err != nil {
				return err
			}

		}
	}
	return nil
}

func FileExists(filename string) bool {
	_, err := os.Stat(filename)
	return err == nil
}
