.PHONY : clean

CC = g++
INCLUDES := -I ../../deps/eigen3.3.4
CPPFLAGS = -fPIC -std=c++11 -Wall -Wno-unknown-pragmas -O3 -fopenmp -fstrict-aliasing -DNDEBUG -Wno-int-in-bool-context
#CPPFLAGS = -fPIC -std=c++11 -g3 -fopenmp
LDFLAGS = 

SOURCES = $(shell echo *.cpp)
HEADERS = $(shell echo *.h)
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = tests

%.o: %.cpp $(HEADERS)
	$(CC) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

all: $(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET)

$(TARGET): $(OBJECTS)
	echo $(OBJECTS)
	$(CC) $(CPPFLAGS) $(INCLUDES) -o $(TARGET) $(OBJECTS) $(LDFLAGS)	 
	
	
