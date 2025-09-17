package main

import (
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"path"
	"strconv"
	"strings"
	"sync"
	"time"
)

type (
	ProcessRecordResponse struct {
		Err error
	}
)

const (
	DATA_IDX_TITLE = 0
	DATA_IDX_TEXT = 1
	DATA_IDX_SUBJECT = 2
	DATA_IDX_DATE = 3
	OUTPUT_IDX_CONTENT = 0
	OUTPUT_IDX_USERNAME = 1
	OUTPUT_IDX_UPLOAD_DATE = 2
	OUTPUT_IDX_CATEGORY = 3
	OUTPUT_IDX_MISINFO = 4
)

var (
	POSSIBLE_DATE_LAYOUTS = [...]string{ "January 2, 2006" }
)

// Output Structure:
// post_content, username, upload_date, category, is_misinformation

func main() {
	cwd, err := os.Getwd()
	if err != nil {
		log.Fatalf("unable to get cwd: %v", err)
	}

	outputPath := path.Join(cwd, "assets", "output.csv")
	outputFile, err := os.Open(outputPath)
	if err != nil {
		log.Fatalf("unable to open %s: %v", outputPath, err)
	}
	defer outputFile.Close()
	var outputWriteMutex sync.Mutex

	trueCsvPath := path.Join(cwd, "assets", "True.csv")

	err = trueDataset(trueCsvPath, outputFile, &outputWriteMutex)
	if err != nil {
		log.Fatalf("unable to parse True.csv: %v", err)
	}

	// fakecsv := path.Join(cwd, "assets", "Fake.csv")
}

func trueDataset(inputFile string, outputFile *os.File, outputWriteMutex *sync.Mutex) error {
	trueFile, err := os.Open(inputFile)
	if err != nil {
		return err
	}
	defer trueFile.Close()

	// Total amount of lines in the CSV file, populated as the file is being read
	lines := 0
	// Number of lines processed. Will be used to calculate when all goroutines are complete
	linesComplete := 0

	trueCsv := csv.NewReader(trueFile)
	recordProcessResponse := make(chan *ProcessRecordResponse)

	for true {
		record, err := trueCsv.Read()
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return err
		}
		lines++
		if lines == 1 {
			continue
		}

		go processRecord(record, false, outputFile, outputWriteMutex, recordProcessResponse)
	}

	for response := range recordProcessResponse {
		if response.Err != nil {
			return response.Err
		}

		linesComplete++;
		if linesComplete >= lines - 1 {
			break
		}
	}

	return nil
}

// Returns a processed, cleaned, and stripped version of the provided
// record ready to be directly inserted into a CSV record. You will 
// have to remember to put in \n
func processRecord(
	record []string, 
	isMisinformation bool,
	outputFile *os.File, 
	outputMutex *sync.Mutex,
	recordProcessResponse chan *ProcessRecordResponse,
) {
	outputRecord := make([]string, 5)
	date, err := stringToDateMultiFormat(trimWhitespace(record[DATA_IDX_DATE]), POSSIBLE_DATE_LAYOUTS[:])
	if err != nil {
		recordProcessResponse <- &ProcessRecordResponse{
			Err: err,
		}
	}

	outputRecord[OUTPUT_IDX_CONTENT] = strings.Trim(record[DATA_IDX_TITLE], " ")
	outputRecord[OUTPUT_IDX_USERNAME] = ""
	outputRecord[OUTPUT_IDX_CATEGORY] = multiStringOp(record[DATA_IDX_SUBJECT], trimWhitespace, strings.ToLower)
	outputRecord[OUTPUT_IDX_MISINFO] = strconv.FormatBool(isMisinformation)
	outputRecord[OUTPUT_IDX_UPLOAD_DATE] = strconv.FormatInt(date.Local().Unix(), 10)

	recordProcessResponse <- &ProcessRecordResponse{
		Err: nil,
	}
}

// Converts a string into a date, that might appear in different layouts
func stringToDateMultiFormat(date string, layouts []string) (time.Time, error) {
	for i := range layouts {
		conv, err := time.Parse(layouts[i], date)
		if err == nil {
			return conv, nil
		}
	}
	return time.Time{}, fmt.Errorf("date '%s' did not match any of the required formats %v", date, layouts)
}

func multiStringOp(str string, operations ...func(string) string) string {
	processed := strings.Clone(str)
	for idx := range operations {
		processed = operations[idx](processed)
	}
	return processed
}

func trimWhitespace(str string) string {
	return strings.Trim(str, " ")
}







