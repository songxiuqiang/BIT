
file(GLOB summer_base_src *.hh base/*.h *.cc)
source_group("" FILES ${summer_base_src})

add_library(summer_base ${summer_base_src})
target_link_libraries(summer_base ${DEP_LIBS})