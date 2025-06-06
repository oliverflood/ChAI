chai_add_executable(IainTinyStories 
    ${CMAKE_CURRENT_SOURCE_DIR}/tinystories.chpl
    ${PROJECT_ROOT_DIR}/lib
)

add_custom_command(TARGET IainTinyStories POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_CURRENT_SOURCE_DIR}/llm"
          "${CMAKE_BINARY_DIR}/tinystories"
  COMMENT "Copying llm folder"
)

set_target_properties(IainTinyStories PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tinystories"
)
