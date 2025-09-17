package data

import (
	"encoding/csv"
	"errors"
	"gomisinfoai/util"
	"io"
	"log"
	"os"
	"path"
	"strconv"
	"strings"
	"sync"
)

var (
	POSSIBLE_DATE_LAYOUTS = [...]string{
		"January 2, 2006",
		"2-Jan-06",
		"Jan 2, 2006",
	}
)

const (
	DATA_IDX_TITLE          = 0
	DATA_IDX_TEXT           = 1
	DATA_IDX_SUBJECT        = 2
	DATA_IDX_DATE           = 3
	OUTPUT_IDX_CONTENT      = 0
	OUTPUT_IDX_USERNAME     = 1
	OUTPUT_IDX_UPLOAD_DATE  = 2
	OUTPUT_IDX_CATEGORY     = 3
	OUTPUT_IDX_MISINFO_TYPE = 4
	OUTPUT_IDX_DATASOURCE   = 5

	MISINFO_TYPE_LIE    = "misinformation"
	MISINFO_TYPE_TRUTH  = "truth"
	MISINFO_TYPE_RUMOUR = "rumour"
)

type (
	ProcessRecordResponse struct {
		LineNumber int
		Err        error
	}
)

func ParseVicUniDataset(
	cwd string,
	outputFileCsv *csv.Writer,
	outputWriteMutex *sync.Mutex,
) {
	trueCsvPath := path.Join(cwd, "assets", "vicuni", "True.csv")
	fakeCsvPath := path.Join(cwd, "assets", "vicuni", "Fake.csv")

	err := processFile(trueCsvPath, outputFileCsv, MISINFO_TYPE_TRUTH, outputWriteMutex)
	if err != nil {
		log.Fatalf("unable to parse True.csv: %v", err)
	}

	err = processFile(fakeCsvPath, outputFileCsv, MISINFO_TYPE_LIE, outputWriteMutex)
	if err != nil {
		log.Fatalf("unable to parse False.csv: %v", err)
	}

}

func processFile(inputFile string, outputFile *csv.Writer, misinfoType string, outputWriteMutex *sync.Mutex) error {
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

		go processVicUniRecord(record, misinfoType, outputFile, outputWriteMutex, recordProcessResponse, lines)
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

func processVicUniRecord(
	record []string,
	misinfoType string,
	outputFile *csv.Writer,
	outputMutex *sync.Mutex,
	recordProcessResponse chan *ProcessRecordResponse,
	lineNumber int,
) {
	outputRecord := make([]string, 6)
	date, err := util.StringToDateMultiFormat(util.TrimWhitespace(record[DATA_IDX_DATE]), POSSIBLE_DATE_LAYOUTS[:])
	if err != nil {
		recordProcessResponse <- &ProcessRecordResponse{
			Err:        err,
			LineNumber: lineNumber,
		}
	}

	outputRecord[OUTPUT_IDX_CONTENT] = util.MultiStringOp(record[DATA_IDX_TITLE], util.TrimWhitespace)
	outputRecord[OUTPUT_IDX_USERNAME] = ""
	outputRecord[OUTPUT_IDX_CATEGORY] = util.MultiStringOp(record[DATA_IDX_SUBJECT], util.TrimWhitespace, strings.ToLower)
	outputRecord[OUTPUT_IDX_MISINFO_TYPE] = misinfoType
	outputRecord[OUTPUT_IDX_UPLOAD_DATE] = util.MultiStringOp(strconv.FormatInt(date.Local().Unix(), 10))
	outputRecord[OUTPUT_IDX_DATASOURCE] = "Victoria University"

	outputMutex.Lock()
	defer outputMutex.Unlock()

	err = outputFile.Write(outputRecord)
	recordProcessResponse <- &ProcessRecordResponse{
		Err: err,
	}
}
