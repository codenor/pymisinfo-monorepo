package main

import (
	"encoding/csv"
	"errors"
	"io"
	"log"
	"os"
	"path"
	"strconv"
	"strings"
	"sync"
)

func ParseMelbourneUni(
	cwd string,
	outputFileCsv *csv.Writer,
	outputWriteMutex *sync.Mutex,
) {
	trueCsvPath := path.Join(cwd, "assets", "melbuni", "True.csv")
	fakeCsvPath := path.Join(cwd, "assets", "melbuni", "Fake.csv")

	err := processFile(trueCsvPath, outputFileCsv, false, outputWriteMutex)
	if err != nil {
		log.Fatalf("unable to parse True.csv: %v", err)
	}

	err = processFile(fakeCsvPath, outputFileCsv, true, outputWriteMutex)
	if err != nil {
		log.Fatalf("unable to parse False.csv: %v", err)
	}

}

func processFile(inputFile string, outputFile *csv.Writer, isMisinformation bool, outputWriteMutex *sync.Mutex) error {
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

		go processRecord(record, isMisinformation, outputFile, outputWriteMutex, recordProcessResponse, lines)
	}

	for response := range recordProcessResponse {
		if response.Err != nil {
			log.Printf("error on line %d (skipping): %v", response.LineNumber, response.Err)
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
	lineNumber int,
) {
	outputRecord := make([]string, 5)
	date, err := stringToDateMultiFormat(trimWhitespace(record[DATA_IDX_DATE]), POSSIBLE_DATE_LAYOUTS[:])
	if err != nil {
		recordProcessResponse <- &ProcessRecordResponse{
			Err:        err,
			LineNumber: lineNumber,
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

