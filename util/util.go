package util

import (
	"fmt"
	"os"
	"strings"
	"time"
)

// Converts a string into a date, that might appear in different layouts
func StringToDateMultiFormat(date string, layouts []string) (time.Time, error) {
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

func TrimWhitespace(str string) string {
	return strings.Trim(str, " ")
}

// Deletes, and re-creates a file to get rid of any existing data
func CleanFile(path string) error {
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
