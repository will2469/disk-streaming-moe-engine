.PHONY: all build validate-m8 test-m8-w1 test-m8-w2 test-m8-w3 test-m8-w4

all: build

build:
	pixi run build

validate-m8:
	pixi run validate-m8

test-m8-w1:
	pixi run test-m8-w1

test-m8-w2:
	pixi run test-m8-w2

test-m8-w3:
	pixi run test-m8-w3

test-m8-w4:
	pixi run test-m8-w4
