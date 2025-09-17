package main

import (
	"strconv"
	"strings"
	"testing"
)

func Test_MultiStringOp(t *testing.T) {
	input := "HELLO"
	expected := "\"hello\""

	actual := MultiStringOp(input, strings.ToLower, strconv.Quote)

	if (expected != actual) {
		t.Errorf("input [%s] does not result in expected [%s], but instead returns [%s]", input, expected, actual)
	}
}

func Test_MultiStringOp_Quotes(t *testing.T) {
	input := "\"HELLO\""
	expected := "\"hello\""

	actual := MultiStringOp(input, strings.ToLower, strconv.Quote)

	if (expected != actual) {
		t.Errorf("input [%s] does not result in expected [%s], but instead returns [%s]", input, expected, actual)
	}
}
