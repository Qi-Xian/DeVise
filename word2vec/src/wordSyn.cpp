#include<vector>
#include<string.h>
#include<string>
#include<math.h>
#include<iostream>
#include<fstream>
#include<stdlib.h>

using namespace std;

const long long max_size = 2000;         // max length of strings
const long long N = 40;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries

string prefix_name,label_name, file_name,train_name;
float *M,len;
char *vocab;
int I;

struct WORD {
 string wnid;
 vector<string> labels;
 vector<int> indexer;
 int size;
};


bool iswnid(string str) {
 int length = str.length();
 int digits = 0;
 if ( length != 9 ) return false;
 for(int i = 1 ; i < length ; i++) {
    if(isdigit(str[i]) != 0) digits++;
 }
 if(digits == 8 && isalpha(str[0]) ) {
  return true;
 } else {
  return false;
 }
}



int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      cout << "Argument missing for "<< str << endl;
      exit(1);
    }
    return a;
  }
  return -1;
}



void saveSyn() {

  int i = 0,nums =0;
  string label;
  fstream input;
  cout <<"Opening : " << label_name << endl;
  input.open(label_name.c_str(),ios::in);
  WORD tmp;
  vector<WORD> WORDS;
    while( input >> label ) {
     if( !iswnid(label) ){
      tmp.labels.push_back(label);
      i++;nums++;
     } else {
      tmp.size = i;
      tmp.wnid = label;
      WORDS.push_back(tmp);
      tmp.labels.clear();
      tmp.size = 0;
      tmp.wnid ="";
      i = 0;
     }
    }

  for(int i = 0 ; i < WORDS.size() ; ++i) 
     WORDS[i].indexer.resize(WORDS[i].size,0);
/*
 *
 * Read & Save words;
 *
 */

  fstream word2vec,savefile;
  cout << "Opening : " << file_name << endl;
  cout << "WORD SIZE: " << nums << endl;
  word2vec.open(file_name.c_str(), ios::in | ios::binary);
  if( word2vec == NULL) {
   cout <<" Input file not found\n";
        return ;
  }
  long long  words,size;
  word2vec >> words; // number of words
  word2vec >> size;  // size of whole vectors
  cout << "words: " << words << endl;
  cout << "vector size: " << size << endl;
  //Allocating memory
  vocab = new char[words * max_w];
  M     = new float[words * size];
  if (M == NULL) {
   cout << "Cannot allocate memory\n";
      return;
  }
 cout <<"Transfering .. \n";
 int ff = 0;
 for (long long  b = 0; b < words; b++) {
   long long a = 0;
   //Transfering vocab
   while (1) {
    word2vec.get(vocab[b*max_w+a]);
    if (word2vec.eof() || (vocab[b * max_w + a] == ' ')) break;
    if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
   }
   vocab[b * max_w + a] = 0;
   //Transfering vectors
   word2vec.read((char*)&M[ b * size], sizeof(float)*size);
   len = 0;
   for (int a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
   len = sqrt(len);
   for (int a = 0; a < size; a++) M[a + b * size] /= len;
   for( int k = 0 ; k < WORDS.size() ; k++) {
    for( int i = 0 ; i < WORDS[k].size ; i++) {
      const char *tmp = WORDS[k].labels[i].c_str();
      if( WORDS[k].indexer[i] != 0 ) continue;
      if ( !strcmp(&vocab[b * max_w],tmp)) {
         savefile.open((prefix_name + WORDS[k].labels[i] + ".bin").c_str(),ios::out|ios::binary);
         savefile << 1 << endl;
         savefile << size << endl;
         savefile << WORDS[k].labels[i] << endl;
         savefile.write((char*)&M[ b*size ],sizeof(float)*size);
         savefile.close();
         WORDS[k].indexer[i] = b;
         nums--;
         ++ff;
     }
    }
   }
   if(nums == 0 ){ cout << "\nall found"; break;}
   if( b%1000 == 0) { 
    cout << "Found " << ff << " words " << b << " files Done.. " << endl;
  }
 }
 word2vec.close();

/*
 *
 * Labeling words;
 *
 */

 fstream vec,debug_;
  cout << "Opening : " << train_name << endl;
  vec.open(train_name.c_str(),ios::out);
  
  for(int k = 0 ; k < WORDS.size(); ++k) {
   for(int m = 0 ; m < WORDS[k].size ; ++m) {
    if( WORDS[k].indexer[m] == 0 ) continue;
    vec << WORDS[k].labels[m] << " " << k << endl; //labels should be start from zero.
   }
  }
  vec.close();
 }
 // end of saveSyn

int main( int argc, char** argv) {

 if(argc < 2 ) {
  cout <<"\n<1>Usage: ./wordSyn <FILE> <LABEL> <TRAIN> <PREFIX>\n \
  where <FILE> contains word projections in the BINARY FORMAT\n \
  where <PREFIX> save location\n \
  where <LABEL> contains file of label\n \
  where <TRAIN> is output for generating train file for CAFFE\n \
  "<< endl;
  return -1;
 }

 if ((I = ArgPos((char *)"-file", argc, argv)) > 0)  file_name =  argv[I + 1];
 if ((I = ArgPos((char *)"-prefix", argc, argv)) > 0)  prefix_name   =  argv[I + 1];
 if ((I = ArgPos((char *)"-label", argc, argv)) > 0)  label_name   =  argv[I + 1];
 if ((I = ArgPos((char *)"-train", argc, argv)) > 0)  train_name   =  argv[I + 1];
 if( argc > 2) saveSyn();
}
                                                            
