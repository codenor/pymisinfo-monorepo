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
	DATA_IDX_TITLE         = 0
	DATA_IDX_TEXT          = 1
	DATA_IDX_SUBJECT       = 2
	DATA_IDX_DATE          = 3
	OUTPUT_IDX_CONTENT     = 0
	OUTPUT_IDX_USERNAME    = 1
	OUTPUT_IDX_UPLOAD_DATE = 2
	OUTPUT_IDX_CATEGORY    = 3
	OUTPUT_IDX_MISINFO     = 4
)

var (
	POSSIBLE_DATE_LAYOUTS = [...]string{"January 2, 2006"}
)

// Output Structure:
// post_content, username, upload_date, category, is_misinformation

func main() {
	cwd, err := os.Getwd()
	if err != nil {
		log.Fatalf("unable to get cwd: %v", err)
	}
	trueCsvPath := path.Join(cwd, "assets", "True.csv")
	outputPath := path.Join(cwd, "assets", "output.csv")

	cleanFile(outputPath)
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
	})
	err = trueDataset(trueCsvPath, outputFileCsv, &outputWriteMutex)
	if err != nil {
		log.Fatalf("unable to parse True.csv: %v", err)
	}

	outputFileCsv.Flush()
	if err := outputFileCsv.Error(); err != nil {
		log.Fatalf("unable to write to output file: %v", err)
	}

	// fakecsv := path.Join(cwd, "assets", "Fake.csv")
}

func trueDataset(inputFile string, outputFile *csv.Writer, outputWriteMutex *sync.Mutex) error {
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
	trueCsv.LazyQuotes = true
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

		linesComplete++
		if linesComplete >= lines-1 {
			break
		}
	}

	return nil
}

func processRecord(
	record []string,
	isMisinformation bool,
	outputFile *csv.Writer,
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

	outputRecord[OUTPUT_IDX_CONTENT] = MultiStringOp(record[DATA_IDX_TITLE], trimWhitespace)
	outputRecord[OUTPUT_IDX_USERNAME] = ""
	outputRecord[OUTPUT_IDX_CATEGORY] = MultiStringOp(record[DATA_IDX_SUBJECT], trimWhitespace, strings.ToLower)
	outputRecord[OUTPUT_IDX_MISINFO] = MultiStringOp(strconv.FormatBool(isMisinformation))
	outputRecord[OUTPUT_IDX_UPLOAD_DATE] = MultiStringOp(strconv.FormatInt(date.Local().Unix(), 10))

	outputMutex.Lock()
	defer outputMutex.Unlock()

	err = outputFile.Write(outputRecord)
	recordProcessResponse <- &ProcessRecordResponse{
		Err: err,
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

func MultiStringOp(str string, operations ...func(string) string) string {
	processed := strings.Clone(str)
	for idx := range operations {
		processed = operations[idx](processed)
	}
	return processed
}

func trimWhitespace(str string) string {
	return strings.Trim(str, " ")
}

// Deletes, and re-creates a file to get rid of any existing data
func cleanFile(path string) error {
	err := os.Remove(path)
	if err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	f.Close()
	return err
}
