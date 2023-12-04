import spacy
from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib3
import math
import pandas as pd
import spacy

## dependencyAndEventCollector collects statistics for events, dependencies of events, dependencies between corresponding events in the same sentence,
## along with POS tag counts, counts of NER types in each sentence, etc. 
class dependencyAndEventCollector:

	## Initialize dictionaries for collecting all unique variables, lists for collecting overall counts per document,
	## initialize the urllib reader to read the websites, spacy class for the english library.  Also read in all the
	## weblinks for use.
	def __init__(self, file):
		self.http = urllib3.PoolManager()
		self.nlp = spacy.load('en_core_web_lg')

		self.verbDicts = {}
		self.depList = {}
		self.idfDict = {}
		self.POS_tag_counts = {}
		self.tokenPairs = {}
		self.argPairs = {}
		self.entCounts = {}

		self.overallVerbCounts = []
		self.overallDepCounts = []
		self.overallIdfCounts = []
		self.overallTokenPairs = []
		self.overallArgPairs = []
		self.overallPOS_tag_counts = []
		self.overallEntCounts = []

		self.listOfVerbs = []

		with open(file) as f:
    			lines = f.read().splitlines()
		self.actors = lines
		

	def tag_visible(self,element):
		if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
			return False
		if isinstance(element, Comment):
			return False
		return True

	## Parse all the text via BeautifulSoup and return all text in the weblink.
	def text_from_html(self,body):
		soup = BeautifulSoup(body, 'html.parser')
		texts = soup.findAll(text=True)
		visible_texts = filter(self.tag_visible, texts)
		return u" ".join(t.strip() for t in visible_texts)

	## Get unique dictionary keys of lemmatized verbs and their given dependency.
	def getVerbDicts_depList(self):
		for actor in self.actors:
       			r = self.http.request('GET', actor)
       			doc = self.nlp(self.text_from_html(r.data))
       			for token in doc:
               			if(token.pos_ == 'VERB'):
                       			if token.lemma_ not in self.verbDicts.keys():
                               			self.verbDicts[token.lemma_] = 0
                       			if (token.lemma_, token.dep_) not in self.depList.keys():
                               			self.depList[(token.lemma_, token.dep_)] = 0

		return self.verbDicts, self.depList

	## Get overall counts per document of lemmatized verbs and associated dependencies.
	def getOverallVerbCounts_DepCounts(self):
		for actor in self.actors:
			self.verbDicts = dict.fromkeys(self.verbDicts, 0)
			self.depList = dict.fromkeys(self.depList, 0)
			r = self.http.request('GET', actor)
			doc = self.nlp(self.text_from_html(r.data))
			for token in doc:
				if(token.pos_ == 'VERB'):
				  if token.lemma_ not in self.verbDicts.keys():
					  self.verbDicts[token.lemma_] = 0
				  else:
					  self.verbDicts[token.lemma_] += 1
		
				  if (token.lemma_, token.dep_) not in self.depList.keys():
					self.depList[(token.lemma_, token.dep_)] = 0
				  else:
					self.depList[(token.lemma_, token.dep_)] += 1

			self.overallVerbCounts.append(list(self.verbDicts.values()))
			self.overallDepCounts.append(list(self.depList.values()))

		return self.overallVerbCounts, self.overallDepCounts

	## Get unique POS tags, lemmatized verb along with specific type of verb (ADVERB and so on), and specific types of NERS.
	def getTokenPairs(self):
		for actor in self.actors:
			r = self.http.request('GET', actor)
			doc = self.nlp(self.text_from_html(r.data))
			for token in doc:
        			if(token.pos_ == 'VERB'):
                			self.listOfVerbs.append(token.lemma_)

			for ent in doc.ents:
				if ent.label_ not in self.entCounts:
					self.entCounts[ent.label_] = 0


			chk_set = set(['VERB'])
			for sent in doc.sents:
        			if (chk_set.issubset(t.pos_ for t in sent) == True):
                			for token in sent:
                        			if token.pos_ not in self.POS_tag_counts:
                               				self.POS_tag_counts[token.pos_] = 0

               				for token in sent:
                       				if (token.lemma_ in self.listOfVerbs):
                               				for key in self.POS_tag_counts:
                                       				if (token.lemma_, key) not in self.tokenPairs.keys():
                                               				self.tokenPairs[(token.lemma_, key)] = 0

		return self.POS_tag_counts, self.tokenPairs, self.entCounts, self.listOfVerbs

	## Get overall count of POS tags, lemmatized verb and verb type and specific types of NERS for each document. 
	def getOverallPairs(self):
		for actor in self.actors:
			self.POS_tag_counts = dict.fromkeys(self.POS_tag_counts, 0)
			self.tokenPairs = dict.fromkeys(self.tokenPairs, 0)
			self.entCounts = dict.fromkeys(self.entCounts, 0)
			r = self.http.request('GET', actor)
			doc = self.nlp(self.text_from_html(r.data))

			for ent in doc.ents:
				self.entCounts[ent.label_] += 1

			chk_set = set(['VERB'])
			for sent in doc.sents:
				if (chk_set.issubset(t.pos_ for t in sent) == True):
					for token in sent:
						self.POS_tag_counts[token.pos_] += 1
				
					for token in sent:
						if (token.lemma_ in self.listOfVerbs):
							for key in self.POS_tag_counts:
								if (token.lemma_, key) in self.tokenPairs.keys():
									self.tokenPairs[(token.lemma_, key)] += 1

			self.overallPOS_tag_counts.append(list(self.POS_tag_counts.values()))
			self.overallTokenPairs.append(list(self.tokenPairs.values()))
			self.overallEntCounts.append(list(self.entCounts.values()))

		return self.overallTokenPairs, self.overallPOS_tag_counts, self.overallEntCounts

	## For each sentence, see if there are more than one events and they are connected by a dependency.  If they are, create a dictionary for the two events along
	## with their connecting dependency.
	def getArgumentPairs(self):
		chk_set = set(['VERB'])
		for actor in self.actors:
			r = self.http.request('GET', actor)
			doc = self.nlp(self.text_from_html(r.data))
			for sent in doc.sents:
				verb1 = ''
				verb2 = ''
				dep1 = ''
				dep2 = ''
				if (chk_set.issubset(t.pos_ for t in sent) == True):
					for token in sent:
						if token.pos_ == 'VERB' and verb1 == '':
							verb1 = token.lemma_
							dep1 = token.dep_
						elif token.pos_ == 'VERB' and verb1 != '':
							verb2 = token.lemma_
							dep2 = token.dep_
						else:
							continue

					if (dep1 == dep2):
						if (verb1, verb2, dep1) not in self.argPairs.keys():
							self.argPairs[(verb1, verb2, dep1)] = 0

		return self.argPairs

	## For each document, get the overall counts of two coreferencing events and their associated dependency.
	def getOverallArgPairs(self):
		chk_set = set(['VERB'])
		for actor in self.actors:
			self.argPairs = dict.fromkeys(self.argPairs, 0)
			r = self.http.request('GET', actor)
			doc = self.nlp(self.text_from_html(r.data))
			for sent in doc.sents:
				verb1 = ''
				verb2 = ''
				dep1 = ''
				dep2 = ''
				if (chk_set.issubset(t.pos_ for t in sent) == True):
					for token in sent:
						if token.pos_ == 'VERB' and verb1 == '':
							verb1 = token.lemma_
							dep1 = token.dep_
						elif token.pos_ == 'VERB' and verb1 != '':
							verb2 = token.lemma_
							dep2 = token.dep_
						else:
							continue

					if (dep1 == dep2):
						if (verb1, verb2, dep1) in self.argPairs.keys():
							self.argPairs[(verb1, verb2, dep1)] += 1


			self.overallArgPairs.append(list(self.argPairs.values()))

		return self.overallArgPairs

