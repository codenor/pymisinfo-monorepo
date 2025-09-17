package main

import (
	"encoding/csv"
	"log"
	"os"
	"path"
	"sync"
)

type (
	ProcessRecordResponse struct {
		LineNumber int
		Err        error
	}
)

const (
	DATA_IDX_TITLE         = 0
	DATA_IDX_TEXT          = 1
	DATA_IDX_SUBJECT       = 2
	DATA_IDX_DATE          = 3
	OUTPUT_IDX_CONTENT     = 0
	OUTPUT_IDX_USERNAME    = 1
	OUTPUT_IDX_UPLOAD_DATE = 2
	OUTPUT_IDX_CATEGORY    = 3
	OUTPUT_IDX_MISINFO     = 4
	OUTPUT_IDX_DATASOURCE  = 5
)

var (
	POSSIBLE_DATE_LAYOUTS = [...]string{
		"January 2, 2006",
		"2-Jan-06",
		"Jan 2, 2006",
	}
)

// Output Structure:

func main() {
	// CONFIGURATION

	cwd, err := os.Getwd()
	if err != nil {
		log.Fatalf("unable to get cwd: %v", err)
	}
	outputPath := path.Join(cwd, "assets", "output.csv")

	cleanFile(outputPath)

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
		"is_misinformation",
		"datasource",
	})

	// DATA PROCESSING

	ParseVicUniDataset(cwd, outputFileCsv, &outputWriteMutex)

	// FINISH

	outputFileCsv.Flush()
	if err := outputFileCsv.Error(); err != nil {
		log.Fatalf("unable to write to output file: %v", err)
	}
}
