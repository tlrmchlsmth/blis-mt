cd ..
make -j 24
make install
cd test
make test_gemm_mpi_blis.x
make test_gemm_mpi_essl.x
#make test_gemm_mpi_skew_a_blis.x
#make test_gemm_mpi_skew_b_blis.x
#make test_gemm_mpi_skew_a_essl.x
#make test_gemm_mpi_skew_b_essl.x
