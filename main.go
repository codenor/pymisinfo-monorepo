package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"path"
)

func main() {
	cwd, err := os.Getwd()
	if err != nil {
		log.Fatalf("unable to get cwd: %v", err)
	}

	outputPath := path.Join(cwd, "assets", "output.csv")
	trueCsvPath := path.Join(cwd, "assets", "True.csv")

	err = trueDataset(trueCsvPath, outputPath)
	if err != nil {
		log.Fatalf("unable to parse True.csv: %v", err)
	}

	// fakecsv := path.Join(cwd, "assets", "Fake.csv")
}

func trueDataset(inputFile string, outputPath string) error {
	trueFile, err := os.Open(inputFile)
	if err != nil {
		return err
	}

	totalLines := 0
	scanner := bufio.NewScanner(trueFile)
	for scanner.Scan() {
		totalLines++;

		if totalLines == 1 {
			continue // skip first line
		}

		fmt.Println(scanner.Text())
		return nil
	}

	if err := scanner.Err(); err != nil {
		return err
	}

	return nil
}
