package data

import (
	"os"
	"path"
	"strings"
	"unicode"
)

var (
	SYMBOLS = []string {
		"?", "!", "...",
	}
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

func IsAllCaps(word string) (bool) {
	for i := range word {
		b := word[i]
		if unicode.IsLower(rune(b)) {
			return false
		}
	}
	return true
}

// Returns the amount of words that are all caps. Discludes words that are 
// only one character long
func CountWordsAllCaps(text string) int {
	totalAllCaps := 0
	words := strings.Split(text, " ")
	for i := range words {
		toCheck := strings.TrimSpace(words[i])
		if len(toCheck) <= 1 {
			continue
		}
		if IsAllCaps(toCheck) {
			totalAllCaps++;
		}
	}
	return totalAllCaps
}
