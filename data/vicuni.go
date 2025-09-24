package data

import (
	"encoding/csv"
	"errors"
	"fmt"
	read "gomisinfoai/readability"
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
	OUTPUT_IDX_HEDGE_CHARS  = 6
	OUTPUT_IDX_SYMBOLS      = 7
	OUTPUT_IDX_ALL_CAPS     = 8
	OUTPUT_IDK_FK_SCORE     = 9

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
	hedgeWords, err := GetHedgeWords(cwd)
	if err != nil {
		log.Fatalf("unable to get hedge words: %v", err)
	}

	trueCsvPath := path.Join(cwd, "assets", "vicuni", "True.csv")
	fakeCsvPath := path.Join(cwd, "assets", "vicuni", "Fake.csv")

	err = processFile(trueCsvPath, outputFileCsv, MISINFO_TYPE_TRUTH, outputWriteMutex, hedgeWords)
	if err != nil {
		log.Fatalf("unable to parse True.csv: %v", err)
	}

	err = processFile(fakeCsvPath, outputFileCsv, MISINFO_TYPE_LIE, outputWriteMutex, hedgeWords)
	if err != nil {
		log.Fatalf("unable to parse False.csv: %v", err)
	}

}

func processFile(
	inputFile string,
	outputFile *csv.Writer,
	misinfoType string,
	outputWriteMutex *sync.Mutex,
	hedgeWords []string,
) error {
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

		go processVicUniRecord(record, misinfoType, outputFile, outputWriteMutex, recordProcessResponse, lines, hedgeWords)
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

// Counts the amount of times a phrase has been mentioned in a string
func countOfCharactersWithinText(text string, phrases []string) int {
	total := 0
	for i := range phrases {
		total += strings.Count(text, phrases[i])
	}
	return total
}

func processVicUniRecord(
	record []string,
	misinfoType string,
	outputFile *csv.Writer,
	outputMutex *sync.Mutex,
	recordProcessResponse chan *ProcessRecordResponse,
	lineNumber int,
	hedgeWords []string,
) {
	outputRecord := make([]string, 10)
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
	outputRecord[OUTPUT_IDX_HEDGE_CHARS] = strconv.FormatInt(int64(countOfCharactersWithinText(record[DATA_IDX_TITLE], hedgeWords)), 10)
	outputRecord[OUTPUT_IDX_SYMBOLS] = strconv.FormatInt(int64(countOfCharactersWithinText(record[DATA_IDX_TITLE], SYMBOLS)), 10)
	outputRecord[OUTPUT_IDX_ALL_CAPS] = strconv.FormatInt(int64(CountWordsAllCaps(record[DATA_IDX_TITLE])), 10)
	outputRecord[OUTPUT_IDK_FK_SCORE] = fmt.Sprintf("%f", read.Fk(record[DATA_IDX_TITLE]))

	log.Printf("%f: %s", read.Fk(record[DATA_IDX_TITLE]), record[DATA_IDX_TITLE])

	outputMutex.Lock()
	defer outputMutex.Unlock()

	err = outputFile.Write(outputRecord)
	recordProcessResponse <- &ProcessRecordResponse{
		Err: err,
	}
}
