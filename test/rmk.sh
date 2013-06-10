cd ..
make -j 16
make install
cd test
make test_gemm_correct_blis.x
