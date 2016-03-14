// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

bool vecToDatum_single( const string&filename, const int label,  Datum* datum , const int CHECKING) {
  LOG(INFO) << filename;  
  std::ifstream word2vec(filename.c_str(),ios::binary);
  string vocab;
  int* words = new int;
  long long* size = new long long;
   
  if( !word2vec.is_open()) {
                 LOG(ERROR) <<" Input file not found\n";
                return false;
   }
  //word2vec.read((char*)words,sizeof(int)); // number of words
  //word2vec.read((char*)size,sizeof(long long));  // size of whole vectors
  word2vec >> *words;
  word2vec >> *size;
  word2vec >> vocab;
  word2vec.get();
  //LOG(INFO) << *words <<" " << *size;
  datum->set_channels(*size);
  datum->set_height(1); 	  // -> row
  datum->set_width( 1);           // -> columns
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  datum->set_label(label);
  int datum_height = datum->height();	//row
  int datum_channel  = datum->channels();//column
  int datum_size   =  datum_height * datum_channel;
  // LOG(INFO) << "Datum size: " << datum_size;
 
  float *vector = new float[datum_size];
  word2vec.read((char*)vector,sizeof(float)*datum_size);
  for (int h = 0; h < *words ; h++) {
    for (int w = 0; w < *size ; w++) {
        datum->add_float_data(vector[w + datum->channels()*h]);     
	if( CHECKING == 1)
	LOG(INFO) << datum->float_data(datum->channels()*h+w);
      }
  }
 // datum->set_data(vector,);
  delete words;
  delete size;
  delete[] vector;
  word2vec.close();
  return true;
}
////////////////////////////////////////////////////////////////////
//
// Saving the set of false text terms to possible image labels
//
///////////////////////////////////////////////////////////////////
bool vecToDatum_multi( const string&filename, const int label,  Datum* datum, const int CHECKING) {
  LOG(INFO) << filename;  
  std::ifstream word2vec(filename.c_str(),ios::binary);
  string vocab;
  int* words = new int;
  long long* size = new long long;
   
  if( !word2vec.is_open()) {
                 LOG(ERROR) <<" Input file not found\n";
                return false;
   }
  word2vec >> *words;
  word2vec >> *size;
  datum->set_channels(*words);
  datum->set_height(*size); 	  // -> row
  datum->set_width(1);           // -> columns
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  datum->set_label(label);
  int datum_height   = datum->height();	
  int datum_channel  = datum->channels();
  int datum_size     = datum_height * datum_channel; 
  //read data from binary file
  float *vector = new float[datum_size];
  for( int i = 0 ; i < *words ; ++i ) {
   word2vec >> vocab; 
   word2vec.get();
   word2vec.read((char*)&vector[i*(*size)],sizeof(float)*(*size));
  }
  for(int h = 0; h < *words ; h++) {
    for(int w = 0; w < *size ; w++) {
        datum->add_float_data(vector[w + datum->height()*h]);     
	if( CHECKING == 1)
	LOG(INFO) << datum->float_data(datum->height()*h+w);
     }
   }
  
  delete words;
  delete size;
  delete[] vector;
  word2vec.close();
  return true;
}




DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");
DEFINE_bool(mode, false,
        "When this option is on, saving word2vec which is restricting the set of false text terms to possible imange labels. ");
DEFINE_int32(CHECKING, 0,
        "When this option is 1, you can check you word2vec file data. ");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of words to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_word2vec [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_word2vec");
    return 1;
  }

  // const bool check_size = FLAGS_check_size;
  //const bool encoded = FLAGS_encoded;

  std::ifstream infile(argv[2]);
  std::vector<std::pair<std::string, int> > lines;
  std::string filename;
  int label;
  while (infile >> filename >> label ) {
    lines.push_back(std::make_pair(filename, label));
  }
  
  LOG(INFO) << "A total of " << lines.size() << " words.";
  LOG(INFO) << "Saving mode is " << FLAGS_mode;
  LOG(INFO) << "CHECKING is " << FLAGS_CHECKING;
  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[3], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // Storing to db
  std::string root_folder(argv[1]);
  Datum datum;
  int count = 0;
  const int kMaxKeyLength = 256;
  const int CHECKING  = FLAGS_CHECKING;
  char key_cstr[kMaxKeyLength];

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    bool status;
   if( !FLAGS_mode ) {  
    status = vecToDatum_single(root_folder + lines[line_id].first +".bin", lines[line_id].second , &datum,CHECKING);
   } else { 
    status = vecToDatum_multi(root_folder + lines[line_id].first +".bin", lines[line_id].second , &datum,CHECKING);
   }
    if (status == false) continue;

    // sequential
    int length = snprintf(key_cstr, kMaxKeyLength, "%08d_%s", line_id,
        lines[line_id].first.c_str());

    // Put in db
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(string(key_cstr, length), out);

    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(ERROR) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(ERROR) << "Processed " << count << " files.";
  }
  return 0;
}
