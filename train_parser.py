#!/usr/bin/env python
import xml.etree.cElementTree as ET
import re
import nltk
import nltk.chunk
import numpy
import random
import nltk.tag
#from nltk.tag import brill_modified
from nltk.tag import brill
from nltk.tag import brill_trainer
#from nltk.etree.ElementTree import ElementTree
import xml.etree
from nltk import Tree
from nltk.tag import pos_tag  
from nltk.tokenize import word_tokenize
import pickle

tags=[]
label=''

def flatten_childtrees(trees):
  children = []
  for t in trees:
    if t.height() < 3:
      children.extend(t.pos())
    elif t.height() == 3:
      children.append(Tree(t.node, t.pos()))
    else:
      children.extend(flatten_childtrees([c for c in t]))
  return children
def flatten_deeptree(tree):
  return Tree(tree.node, flatten_childtrees([c for c in tree]))	
def replacer(filename):
	exml=nltk.data.find(filename)
	raw=open(exml).read()
	raw2=raw.replace('<ALL>', '')
	raws=raw2.replace('</ALL>', '')
	descs=[]
	delim="</S>"
	descs=[e+delim for e in raws.split(delim)]
	descs.pop()
	print len(descs)
	trees=[]
	fulltrees=[]
	pattern=r'<(?!/)'
	pattern_close=r'<\/[A-Za-z]*>'
	pattern_ending=r'>'
	for i, d in enumerate(descs):
		c=d
		c=re.sub(pattern, '(', c)
		c=re.sub(pattern_close, ')', c)
		c=re.sub(pattern_ending, ' ', c)
		#print c
		fulltrees.append(nltk.Tree(c))
		trees.append(flatten_deeptree(nltk.Tree(c)))
	#print trees
	return trees, fulltrees
def training_testing_replacer(filename, percent_training):
	exml=nltk.data.find(filename)
	raw=open(exml).read()
	raw2=raw.replace('<ALL>', '')
	raws=raw2.replace('</ALL>', '')
	descs=[]
	delim="</S>"
	descs=[e+delim for e in raws.split(delim)]
	descs.pop()
	print len(descs)
	nums=list(xrange(0, len(descs), 1))
	training_count=int(round(percent_training*len(descs)))
	training=random.sample(nums, training_count)
	training.sort()
	testing=list(set(nums)-set(training))
	fulltrees_train=[]
	fulltrees_test=[]
	pattern=r'<(?!/)'
	pattern_close=r'<\/[A-Za-z]*>'
	pattern_ending=r'>'

	for i, d in enumerate(descs):
		c=d
		c=re.sub(pattern, '(', c)
		c=re.sub(pattern_close, ')', c)
		c=re.sub(pattern_ending, ' ', c)
		#print i
		#print c
		if i in training:
			fulltrees_train.append(nltk.Tree.fromstring(c))
		elif i in testing:
			fulltrees_test.append(nltk.Tree.fromstring(c))
		else:
			print "error"
		#trees.append(flatten_deeptree(nltk.Tree(c)))
	#print trees
	return fulltrees_train, fulltrees_test

def initial_unigram_chunker(words_train, tags_train, flag='ubt'):
	word_u_chunker=nltk.tag.UnigramTagger(words_train)
	tag_u_chunker=nltk.tag.UnigramTagger(tags_train)
	word_b_chunker=nltk.tag.BigramTagger(words_train, backoff=word_u_chunker)
	tag_b_chunker=nltk.tag.BigramTagger(tags_train, backoff=tag_u_chunker)
	word_t_chunker=nltk.tag.TrigramTagger(words_train, backoff=word_b_chunker)
	tag_t_chunker=nltk.tag.TrigramTagger(tags_train, backoff=tag_b_chunker)
	word_q_chunker=nltk.tag.NgramTagger(4, words_train, backoff=word_t_chunker)
	tag_q_chunker=nltk.tag.NgramTagger(4, tags_train, backoff=tag_t_chunker)
	word_f_chunker=nltk.tag.NgramTagger(5, words_train, backoff=word_q_chunker)
	tag_f_chunker=nltk.tag.NgramTagger(5, tags_train, backoff=tag_q_chunker)
	print tag_f_chunker.evaluate(tags_train)
##	print word_f_chunker
##	print tag_f_chunker
	return word_f_chunker, tag_f_chunker

def node_is(node):

	pattern = r'[A-Z]{4,5}' #chunk tags
	pattern2 = r'[A-Z]{2,3}' #POS Tags
	if re.match(pattern, node):
		return 'chunk'
	elif re.match(pattern2, node):
		return 'tag'
	else:
		return 'nin'

def nested_tree_to_iob2(t):
	global tags # holds a list of IOB2 tags, triples like [(triple), ('and', 'CC', 'O'), etc]
	global label
	for child in t:
		#print child
		#print "==============="
		try:
			child.label()
			if node_is(child.label())=='chunk':
				#print child.label()+' is a chunk'
				if child.height()>=5:
					label=child.label()+'-'
				else:
					label+=child.label()+'-'
				nested_tree_to_iob2(child)
			elif node_is(child.label())=='tag':
				#print child.label()+' is a tag'
				#print child.leaves()[0]+' is the word'
				tags.append((child.leaves()[0], child.label(), label))
				#print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"		
		except:
			#tags.append((child.leaves()[0], child.label(), label))
			print child.label()
			#print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
			#print child.leaves()
		#print child.label()
		#print "==============="
		#print child.leaves()
		#print "================"
	return tags	
def get_max_height(trees):
	max_height=0;
	max_height_tree=()
	for tree in trees:
		if tree.height()>max_height:
			max_height=tree.height()
			max_height_tree=tree
	return max_height, max_height_tree
def nested_tree_tags(tree):
	tags=[]
	for child in tree:
		if node_is(child.label()) == 'tag':
			tags.append((child.leaves()[0], child.label(), 'O')) #outside every chunk
		elif node_is(child.label())=='chunk':
			for cchild in child:
				if node_is(cchild.label())=='tag':
					tags.append((cchild.leaves()[0], cchild.label(), child.label())) #inside top chunk 
				elif node_is(cchild.label())=='chunk':
					for ccchild in cchild:
						if node_is(ccchild.label())=='tag':
							tags.append((ccchild.leaves()[0], ccchild.label(), child.label()+'-'+cchild.label())) #inside first nest
						elif node_is(ccchild.label())=='chunk':
							for cccchild in ccchild:
								if node_is(cccchild.label())=='tag':
									tags.append((cccchild.leaves()[0], cccchild.label(), child.label()+'-'+cchild.label()+'-'+ccchild.label()))
								elif node_is(cccchild.label())=='chunk':
									for ccccchild in cccchild:
										if node_is(ccccchild.label())=='tag':
											tags.append((ccccchild.leaves()[0], ccccchild.label(), child.label()+'-'+cchild.label()+'-'+ccchild.label()+'-'+cccchild.label()))
										elif node_is(ccccchild.label())=='chunk':
											for cccccchild in ccccchild:
												if node_is(cccccchild.label())=='tag':
													tags.append((cccccchild.leaves()[0], cccccchild.label(), child.label()+'-'+cchild.label()+'-'+ccchild.label()+'-'+cccchild.label()+'-'+ccccchild.label()))
												else: 
													print 'too deeply nested for us'
	return tags
def nested_tree_tags_recursive(tree, subtree):
	tags=[]
	for child in tree:
		if node_is(child.label())=='tag':
			tags.append((child.leaves()[0], child.label(), 'O'))
	return tags
def nested_tag_chunks(chunk_sents):
    tag_sents = [nested_tree_tags(tree) for tree in chunk_sents]
    return [[(t, c) for (w, t, c) in chunk_tags] for chunk_tags in tag_sents]
def nested_word_chunks(chunk_sents):
	for i, tree  in enumerate(chunk_sents):
		nested_tree_tags(tree)
	tag_sents = [nested_tree_tags(tree) for tree in chunk_sents]
	return [[(w, c) for (w, t, c) in chunk_tags] for chunk_tags in tag_sents]	
			
def chunk_word(cmdstr):
    thefile = "./Fixedunique149.xml"
    percent_training = 0.9
##    cmdstr = "Go forward and turn right and go straight until you are at the wall and you will find the laptop on the table with the chairs"
#def main_function(training_file, test_file):
	#[tr_trees, tr_fulltrees]=replacer(training_file)
	#[testing, test_fulltrees]=replacer(test_file)
    global tags
    #templates = [
        #brill.SymmetricProximateTokensTemplate(brill.ProximateWordsRule, (1,1)), #1 or -1 tag
        #brill.SymmetricProximateTokensTemplate(brill.ProximateWordsRule, (2,2)), # 2 or -2 tag
        #brill.SymmetricProximateTokensTemplate(brill.ProximateWordsRule, (1,2)), #1 to 2 or -1 to -2 tag
        #brill.SymmetricProximateTokensTemplate(brill.ProximateWordsRule, (1,3)), # 1 to 3 or -1 to -3 tag
        #brill.SymmetricProximateTokensTemplate(brill.ProximateWordsRule, (1,5)), # 1 to 7 or -1 to -7 tag
        #brill.SymmetricProximateTokensTemplate(brill.ProximateWordsRule, (1,4)), # 1 to 7 or -1 to -7 tag
        #brill.SymmetricProximateTokensTemplate(brill.ProximateWordsRule, (1,6)), # 1 to 7 or -1 to -7 tag
        #brill.SymmetricProximateTokensTemplate(brill.ProximateWordsRule, (1,7)), # 1 to 7 or -1 to -7 tag
        #brill.SymmetricProximateTokensTemplate(brill.ProximateWordsRule, (1,8)), # 1 to 7 or -1 to -7 tag
        #brill.SymmetricProximateTokensTemplate(brill.ProximateWordsRule, (1,9)), # 1 to 7 or -1 to -7 tag
        #brill.SymmetricProximateTokensTemplate(brill.ProximateWordsRule, (1,10)), # 1 to 7 or -1 to -7 tag
        #brill.SymmetricProximateTokensTemplate(brill.ProximateTagsRule, (1,1)), #-1 interval or 1 interval chunk
        #brill.SymmetricProximateTokensTemplate(brill.ProximateTagsRule, (2,2)),	# -2 interval or 2 interval chunk
        #brill.SymmetricProximateTokensTemplate(brill.ProximateTagsRule, (1,2)), # 1 to 2 or -1 to -2 chunk
        #brill.SymmetricProximateTokensTemplate(brill.ProximateTagsRule, (1,3)), #1 to 3 or -1 to -3 chunk
        #brill.SymmetricProximateTokensTemplate(brill.ProximateTagsRule, (1,5)), # 1 to 7 or -1 to -7 tag
        #brill.SymmetricProximateTokensTemplate(brill.ProximateTagsRule, (1,4)), # 1 to 7 or -1 to -7 tag
        #brill.SymmetricProximateTokensTemplate(brill.ProximateTagsRule, (1,6)), # 1 to 7 or -1 to -7 tag
        #brill.SymmetricProximateTokensTemplate(brill.ProximateTagsRule, (1,7)), # 1 to 7 or -1 to -7 tag
        #brill.SymmetricProximateTokensTemplate(brill.ProximateTagsRule, (1,8)), # 1 to 7 or -1 to -7 tag
        #brill.SymmetricProximateTokensTemplate(brill.ProximateTagsRule, (1,9)), # 1 to 7 or -1 to -7 tag
        #brill.SymmetricProximateTokensTemplate(brill.ProximateTagsRule, (1,10)), # 1 to 7 or -1 to -7 tag
        #brill.ProximateTokensTemplate(brill.ProximateTagsRule, (-1, -1), (1,1)), #-1 and 1 chunk
        #brill.ProximateTokensTemplate(brill.ProximateWordsRule, (-1, -1), (1,1)), #-1 and 1 tag
        #brill.ProximateTokensTemplate(brill.ProximateTagsRule, (-2, -2), (2, 2)), #-2 and 2 chunk
        #brill.ProximateTokensTemplate(brill.ProximateWordsRule, (-2, -2), (2,2)),  #-2 and 2 tag
        #brill.ProximateTokensTemplate(brill.ProximateTagsRule, (-3, -3), (3, 3)), #-2 and 2 chunk
        #brill.ProximateTokensTemplate(brill.ProximateWordsRule, (-3, -3), (3,3)),  #-2 and 2 tag
        #brill.ProximateTokensTemplate(brill.ProximateTagsRule, (-4, -4), (4, 4)), #-2 and 2 chunk
        #brill.ProximateTokensTemplate(brill.ProximateWordsRule, (-4, -4), (4,4)),  #-2 and 2 tag
        #brill.ProximateTokensTemplate(brill.ProximateTagsRule, (-5, -5), (5, 5)), #-2 and 2 chunk
        #brill.ProximateTokensTemplate(brill.ProximateWordsRule, (-5, -5), (5,5))  #-2 and 2 tag
        #]

    #print t_fulltrees[0]
    #print testing[0]
    [tr_fulltrees, test_fulltrees] = training_testing_replacer(thefile, percent_training)
    print '\n'
    #traverse(t_fulltrees[55])
    tags = []
    tags_test = nested_tag_chunks(test_fulltrees)
    tags = []
    words_test = nested_word_chunks(test_fulltrees)
    tags = []
    words_train = nested_word_chunks(tr_fulltrees)
    tags = []
    tags_train = nested_tag_chunks(tr_fulltrees)
    tags = []
    [bls0, bls1] = initial_unigram_chunker(words_train, tags_train)
    trainer_tags = brill_trainer.BrillTaggerTrainer(bls1, nltk.tag.brill.brill24(), trace=3)
    brill_chunker_tags = trainer_tags.train(tags_train, max_rules=450, min_score=4)
    trainer_words = brill_trainer.BrillTaggerTrainer(bls0, nltk.tag.brill.brill24(), trace=3)
    brill_chunker_words = trainer_words.train(words_train, max_rules=450, min_score=4)
##    print 'Trigram on words training'
##    print bls0.evaluate(words_train)
##    print 'Trigram on words testing'
##    print bls0.evaluate(words_test)
##    print 'Trigram on POS tags training'
##    print bls1.evaluate(tags_train)
##    print 'Trigram on POS tags testing'
##    print bls1.evaluate(tags_test)
##    print 'Brill POS tags testing'
##    print brill_chunker_tags.evaluate(tags_test)
##    print 'Brill POS tags training'
##    print brill_chunker_tags.evaluate(tags_train)
##    print 'Brill words testing'
##    print brill_chunker_words.evaluate(words_test)
##    print 'Brill words training'
##    print brill_chunker_words.evaluate(words_train)
    cmdlist = cmdstr.split()
    print brill_chunker_words.tag(cmdlist)
    pickle.dump( brill_chunker_words, open( "./brill_chunker_words.p", "wb" ) )
    return brill_chunker_words.tag(cmdlist)

def chunk_word_directly(cmdstr):
	brill_chunker_words = pickle.load( open( "./brill_chunker_words.p", "rb" ) )
	cmdlist = cmdstr.split()
	return brill_chunker_words.tag(cmdlist)

##raw to tag
def raw_to_tree(cmdstr):
    raw = cmdstr
##    raw = "Go forward and turn right and go straight until you are at the wall and you will find the laptop on the table with the chairs"
    tags = pos_tag(word_tokenize(raw))
    ##chunker = [('Go', None), ('forward', 'ORMTP'), ('and', 'ORMTP'), ('turn', 'ORMTP-ORMRP'), ('right', 'ORMTP-ORMRP'), ('and', 'O'), ('go', 'IRMRP'), ('straight', 'IRMRP'),
    ##           ('until', 'IRMRP'), ('you', 'IRMRP'), ('are', 'IRMRP'), ('at', 'IRMRP'), ('the', 'IRMRP'), ('wall', 'IRMRP'), ('and', 'O'),
    ##           ('you', 'OBTP'), ('will', 'OBTP'), ('find', 'OBTP'), ('the', 'OBTP'), ('laptop', 'OBTP'), ('on', 'OBTP-FURTP'), ('the', 'OBTP-FURTP'),
    ##           ('table', 'OBTP-FURTP'), ('with', 'OBTP-FURTP-FURRP'), ('the', 'OBTP-FURTP-FURRP'), ('chairs', 'OBTP-FURTP-FURRP')]
    chunker = chunk_word_directly(cmdstr)
    for l in chunker:
        r = list(l)
        chunker[chunker.index(l)] = r

    print chunker

    for l in chunker:
        text = l[0]
        label = l[1]
        print text, label
        if type(label) is str:
            chunks = label.split('-')
            chunker[chunker.index(l)].append(chunks)
        else:
            chunker[chunker.index(l)].append([])

    L = len(chunker)
    leaves = [[]] * L
    for i in range(L):
        leaves[i] = Tree(tags[i][1], [chunker[i][0]])

    node = []
    hg = [0] * L
    for i in range(L):
        hg[i] = len(chunker[i][2])

    for i in range(L):
        if (i != 0 and (hg[i] != hg[i-1] or chunker[i][2] != chunker[i-1][2])) or i == 0:
            children = []
            for j in range(i, L):
                if hg[j] < hg[i] or chunker[j][2] != chunker[i][2]:
                    break
                else:
                    if chunker[j][2] == chunker[i][2]:
                        children.append(leaves[j])
            name = chunker[i][2]
            print i
            if len(name) > 0:
                node.append([i, name[len(name)-1], hg[i], children])
            else:
                node.append([i, [], hg[i], children])
    print "_____________________________"
    for n in node:
        print n
    print "_____________________________"

    H = max(hg)
    node.insert(0, [0, "S", 0, Tree('S', [])])

    for hi in range(H):
        h = H - hi
        print "h", h
        NL = len(node)
        print "node len", NL
        print node
        i = 0
        while i < NL:
            height = node[i][2]
            print i, height
            if height == h:
                print i
                node[i-1][3].append(Tree(node[i][1], node[i][3]))
                del node[i]
                NL = len(node)
            else:
                i = i+1

    return node[0][3]

def show_height(tree):
    H = tree.height()
    treepos = tree.treepositions()
    heightset = []
    
    for s in treepos:
        heightset.append(len(s))
##        print tree[s]
##    print heightset

    return heightset

def show_leave(tree):
    heightset = show_height(tree)
    res = [0] * len(heightset)
##    print "rse", res
    for i in range(len(heightset)-1):
        if heightset[i] > heightset[i+1]:
            res[i] = 1
    res[len(heightset)-1] = 1
    return res

def raw_to_xml(cmdstr, filename):
    tree = raw_to_tree(cmdstr)
##    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
##    tree.draw()
    Height = tree.height()
    treepos = tree.treepositions()
##    print Height, treepos
    NODE = [[]]*Height
##    print NODE
    L = len(treepos)

    ALL = ET.Element("ALL")
    heightset = show_height(tree)
    leaveset = show_leave(tree)
##    print leaveset

    level = 0
    idx = 0
    node = [[]]*L
    node[0] = ET.SubElement(ALL, "S")
    
    for i in treepos:
        idx = treepos.index(i)
##        print "i:", idx
        idx2 = idx
        hgt = len(i)
##        print node
##        print "node idx: ", node[idx]
        for j in range(idx+1, L):
            if heightset[j] <= hgt or j == L-1:
                idx2 = j-1
                if j == L-1:
                    idx2 = L - 1
                break
##        print idx+1, idx2
        subnodeset = []
        for p in range(idx+1, idx2+1):
            if heightset[p] == hgt + 1:
                subtext = ""
                if leaveset[p] == 0:
                    subtext = tree[treepos[p]].label()
                    print subtext
                    if '$' in subtext:
			subtext = subtext.replace('$', '');
                    node[p] = ET.SubElement(node[idx], subtext)
                else:
                    subtext = tree[treepos[p]]
                    node[idx].text = subtext
##                print idx, p, subtext
                

    xmltree = ET.ElementTree(ALL)
    xmltree.write(filename)
    return tree


if __name__ == '__main__':
    cmdstr = "the fork will be on the table to the right"
    filename = "./Fixedunique149.xml"
    chunker = chunk_word(cmdstr)
    tree = raw_to_xml(cmdstr, filename)
    tree.draw()
