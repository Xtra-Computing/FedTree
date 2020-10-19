set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

file(GLOB SRC util/*.c* FL/*.c* tree/*.c* *.c*)
list(REMOVE_ITEM SRC "${CMAKE_CURRENT_LIST_DIR}/main.cpp")

cuda_add_library(${PROJECT_NAME} SHARED ${SRC})
if (USE_CUDA)
target_link_libraries(${PROJECT_NAME} ${CUDA_cusparse_LIBRARY})
endif ()

cuda_add_executable(${PROJECT_NAME}-train main.cpp)
target_link_libraries(${PROJECT_NAME}-train ${PROJECT_NAME})

