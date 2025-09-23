package data

import (
	"os"
	"path"
	"strings"
)

func GetHedgeWords(cwd string) ([]string, error) {
	fp := path.Join(cwd, "assets", "hedge-words.txt")
	b, err := os.ReadFile(fp)
	if err != nil {
		return nil, err
	}
	
	wordStr := string(b)
	return strings.Split(wordStr, " "), nil
}
