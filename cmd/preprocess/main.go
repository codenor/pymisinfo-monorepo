package main

import (
	"encoding/csv"
	"gomisinfoai/data"
	"gomisinfoai/util"
	"log"
	"os"
	"path"
	"sync"
)

// Output Structure:

func main() {
	// CONFIGURATION

	cwd, err := os.Getwd()
	if err != nil {
		log.Fatalf("unable to get cwd: %v", err)
	}
	outputPath := path.Join(cwd, "assets", "output.csv")

	util.CleanFile(outputPath)

	// OPEN OUTPUT FILE

	outputFile, err := os.OpenFile(outputPath, os.O_APPEND|os.O_WRONLY, os.ModeAppend)
	if err != nil {
		log.Fatalf("unable to open %s: %v", outputPath, err)
	}
	defer outputFile.Close()
	outputFileCsv := csv.NewWriter(outputFile)

	var outputWriteMutex sync.Mutex

	outputFileCsv.Write([]string{
		"content",
		"username",
		"upload_date",
		"category",
		"misinformation_type",
		"datasource",
	})

	// DATA PROCESSING

	data.ParseVicUniDataset(cwd, outputFileCsv, &outputWriteMutex)

	// FINISH

	outputFileCsv.Flush()
	if err := outputFileCsv.Error(); err != nil {
		log.Fatalf("unable to write to output file: %v", err)
	}
}
