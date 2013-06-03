cd ..
make -j
make install
cd test
make test_gemm_correct_blis.x
