package main

import (
	"encoding/csv"
	"errors"
	"io"
	"log"
	"os"
	"path"
	"sync"
)

type (
	ProcessRecordResponse struct {
		Err error
	}
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

		go processRecord(record, outputFile, outputWriteMutex, recordProcessResponse)
	}

	for response := range recordProcessResponse {
		if response.Err != nil {
			return response.Err
		}

		linesComplete++;
		log.Printf("Processed line %d, %d total (%.2f)", linesComplete, lines, float64(linesComplete) / float64(lines))
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
	outputFile *os.File, 
	outputMutex *sync.Mutex,
	recordProcessResponse chan *ProcessRecordResponse,
) {
	recordProcessResponse <- &ProcessRecordResponse{
		Err: nil,
	}
}









