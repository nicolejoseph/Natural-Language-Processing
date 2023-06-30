# NLP Project 2: Parser

# Helpful sources:
# https://medium.com/swlh/cyk-cky-f63e347cf9b4
# https://www.borealisai.com/en/blog/tutorial-15-parsing-i-context-free-grammars-and-cyk-algorithm/

import string 
import sys
from time import process_time
# This module provides regular expression matching operations similar to those found in Perl
import re 

# if "re" not in dir():
    # print("re not imported!")

# node class to keep track of backpointers and store data for non-terminal symbols
class Node:
	def __init__(self, symbol, leftChild, rightChild = None): 
		self.symbol = symbol 
		self.leftChild = leftChild 
		self.rightChild = rightChild 

t1 = process_time()
CNFfileName = input("Enter the name of the text file specifying a CFG in CNF: ")
CNFfile = open(CNFfileName, 'r')

# function to read the CNF file's grammar rules line by line
def readGrammar(CNFfile):
	dictCNF = {}
	while True:
		temp = CNFfile.readline()
		line = temp.rstrip() 
		if not line: 
			break
		count = 0
		index = [] 
		for i in range(len(line)):
			if line[i] == ' ':
				index.append(i)
				count += 1

		# CNF grammar rule has the form A --> w
		# where A is nonterminal and w is terminal		
		if count == 2: 
			A = line[0:index[0]]
			w = line[index[1] + 1:]
			if A in dictCNF.keys(): 
				dictCNF[A].append(w)
			else: 
				dictCNF[A] = [w]

		# A --> B C		
		if count == 3: 
			A = line[0:index[0]]
			B = line[index[1] + 1:index[2]]
			C = line[index[2] + 1:]
			if A in dictCNF.keys(): 
				dictCNF[A].append((B, C))
			else: 
				dictCNF[A] = [(B, C)]

	return dictCNF

dictCNF = readGrammar(CNFfile)

t2 = process_time()
processTime = t2 - t1

#Function to start filling in the table for the CKY algorithm
# Parsing powerpoint slide 25
def diagonalCells(words, dictCNF, table, j):
	for A in dictCNF:
			if words[j - 1] in dictCNF[A]: 
				table[j - 1][j].append(Node(A, words[j - 1])) 
	return table

def otherCells(dictCNF, table, i, j, k):
	listB = table[i][k] 
	listC = table[k][j] 
	if len(listB) > 0 and len(listC) > 0: 
		for A in dictCNF.keys():
			for BNode in listB:
				for CNode in listC:
					if (BNode.symbol, CNode.symbol) in dictCNF[A]: 
						table[i][j].append(Node(A, BNode, CNode))
	return table

def CKY(words, dictCNF, n):
	table = [[[] for col in range(n + 1)] for row in range(n)] 
	for j in range(1, n + 1): 
		diagonalCells(words, dictCNF, table, j)
		for i in range(j - 2, -1, -1): 
			for k in range(i + 1, j): 
				otherCells(dictCNF, table, i, j, k)
	return table

# function to output the parse in bracketed notation
def bracketedNotation(node):
	if not (node.rightChild == None): 
		return "[" + node.symbol + " " +  bracketedNotation(node.leftChild) + " " + bracketedNotation(node.rightChild) + "]"
	else: 
		return "[" + node.symbol + " " + node.leftChild + "]"

# function to display the optional textual parse tree
def printTree(convertBracket):
	tabs = 1 
	prevIndex = 1 
	print('[', end = '') 

	for i in range(1, len(convertBracket)): 
		if convertBracket[i] == '[': 
			print(convertBracket[prevIndex:i] + '\n' + tabs * '\t', end = '')
			tabs += 1
			prevIndex = i
		if convertBracket[i] == ']': 
			print(convertBracket[prevIndex:i + 1] + '\n' + (tabs - 2) * '\t', end = '')
			tabs -= 1
			prevIndex = i + 1

while True: 
	parseTree = input("Do you want the textual parse trees to be displayed (y/n)?: ")
	if parseTree == "quit":
		break
	sentence = input("Enter a sentence: ")
	
	if sentence == "quit":
		break
	
	# convert sentence to lowercase and disregard punctuation
	sentence = sentence.lower() 
	words = re.findall('[A-Za-z0-9]+', sentence) 
	n = len(words)
	table = CKY(words, dictCNF, n)

	parseList = [] 
	parseNum = 0
	for value in table[0][n]:
		if value.symbol == 'S':
			parseList.append(value)
			parseNum += 1

	if parseNum > 0:
		print("VALID SENTENCE\n")
	else:
		print("NO VALID PARSES\n")

	count = 0
	while count < parseNum:
		for node in parseList:
			count += 1
			print(f"Valid parse #{count}: ")
			convertBracket = bracketedNotation(node)
			print(convertBracket + '\n')
			if parseTree == 'y':
				printTree(convertBracket)

	print(f"\nTotal number of valid parses: {parseNum}\n")

CNFfile.close()
