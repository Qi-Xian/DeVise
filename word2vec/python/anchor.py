
from xml.dom import minidom as xmlparser

# dir which contain wordnet structure
#	and wnidlistfile 
s_path ='/v2/gtlim/util/'
w_path ='/'

#name of root in structure xml
root_name = 'fall11'

class Node:
	# root level is 0
	def __init__(self):
		self.level = 1
		self.childs = []
		self.parents = []
		self.nchild = 0
		self.path = []

	def __init__(self,wnid,word):
		self.level = 1
		self.childs = []
		self.parents = []
		self.nchild = 0
		self.wnid = wnid
		self.setnc = True
		self.word = word
		self.setoc = True
		self.path = []


	def set_wnid(self,wnid):
		self.wnid = wnid
		self.setnc = True
	def get_wnid(self):
		if setnc:
			return self.word
		else: 
			return 'error'

	def set_word(self,word):
		self.word = word
		self.setoc = True
	def get_word(self):
		if setoc:
			return self.word
		else: 
			return 'error'

	def append_child(self,child):
		self.childs.append(child)
		self.nchild+=1
	def append_parent(self,parent):
		self.parents.append(parent)
		self.level+=1
	def PathtoRoot(self,index):
			self.path.append(index)

	def getLevel(self):
		return self.level
		

class Anchor:
	def __init__(self,):
		import Node
		# parse structure xmlfile
		taxonomyfile = s_path + 'structure_released.xml'
	        self.doc = xmlparser.parse(taxonomyfile)
		self.xDoc = doc.getElementsByTagName('synset')
		
		#read wnids
		filname = w_path + fin
		f = open(filename,'r')
		self.Nodes = []
		while True:
			title = f.readline()	
			if not title: break
			compo = title.split(':')
			wnid = compo[0].rstrip()
			words = compo[1].split(',')
			new_node = Node(wnid,words[0])
			Nodes.append(new_node)
		logger.info("total #%i nodes" %len(Nodes))
			
			
	def buildTaxonomy(self):	
		logger.info("building a taxonomy ")
		num = len(Nodes)
		#found
		ff = 0
		self.maxlevel = 0;
		for no_no,s_no in enumerate(self.xDoc):
			
			wnid = s_no.getAttribute('wnid').encode()
			words = s_no.getAttribute('words').encode()
			if( wnid == root_name):continue
			
			for no in self.Nodes:
				if( wnid == no.get_wnid()): 
					self.getInfo(s_no,no)
					ff+=1
			if( no_no%1000 == 0):
				logger.info(" Progress at #%i node , #%i nodes were found ", %(no_no,ff))

	#this function will get information of one node 
	def getInfo(self,x,node):
		#first get childnodes
		for no,c_node in enumerate(x.childNodes)
			if no == 1:continue
			new_node = Node(c_node.getAttribute('wnid'),c_node.getAttribute('words').split(',')[0])
			node.append_child(new_node)
				
		while True:
			x = x.parentNode()
			if( x.getAttribute('wnid') == root_name): break
			new_node = Node(x.getAttribute('wnid'),x.getAttribute('words').split(',')[0])
			node.append_parent(new_node)

		self.maxlevel = max(self.maxlevel,node.getLevel())
		
	def mergenode(self,)
	def prunenode(self,


















