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

DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");
DEFINE_bool(verbose, false,
        "When this option is 1, you can check you word2vec file data. ");


int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Create a set of words to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    create_word2vec [FLAGS]  ROOTFOLDER/ BINFILE DB_NAME\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/create_word2vec");
    return 1;
  }

  // const bool check_size = FLAGS_check_size;
  //const bool encoded = FLAGS_encoded;

  std::string filename;  
  LOG(INFO) << "verbose " << FLAGS_verbose;
  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[3], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // Storing to db
  std::string root_folder(argv[1]);
  Datum datum;
  int count = 0;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  long long words,size;
  std::fstream word2vec;
  LOG(INFO) << "Opening : " << argv[2];
  word2vec.open(argv[2], std::ios::in | std::ios::binary);
  if( word2vec == NULL) {
   LOG(INFO) <<"Input file not found\n";
   exit (EXIT_FAILURE);
  }
  word2vec >> words; // number of words
  word2vec >> size;  // size of whole vectors
  LOG(INFO) << "words: " << words;
  LOG(INFO) << "vector size: " << size;
  //Allocating memory
  LOG(INFO)<<"Transfering .. \n";
  std::string vocab;
  float *vector = new float[size];
  datum.set_channels(size);
  datum.set_height(1); 
  datum.set_width(1);  
 for (long long  b = 0; b < words; b++) {

    word2vec >> vocab;
    word2vec.get();
    word2vec.read((char*)vector,sizeof(float)*(size));
    datum.clear_data();
    datum.clear_float_data();
    datum.set_encoded(false);
    datum.set_label(b);
 
    for(int w = 0; w < size ; w++) {
      datum.add_float_data(vector[w]);
       if(FLAGS_verbose == 1)
          LOG(INFO) << datum.float_data(w);
     }
    // sequential
    int length = snprintf(key_cstr, kMaxKeyLength, "%08d_%s", count,
        vocab.c_str());

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
  delete []vector;
  return 0;
}
