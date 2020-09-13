//   Copyright Naoki Shibata and contributors 2010 - 2020.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <stdbool.h>

#define N 1000

int nkeywords = 0, nalloc = 0;
char **keywords = NULL, *suffix = NULL;

void insert(char *buf) {
  for(int i=0;i<nkeywords;i++) {
    if (strcmp(keywords[i], buf) == 0) printf("%s", suffix);
  }
}

void doit(FILE *fp) {
  int state = 0;
  bool nl = true;
  char buf[N+10], *p = buf;

  for(;;) {
    int c = getc(fp);
    if (c == EOF) break;
    switch(state) {
    case 0:
      if (isalnum(c) || c == '_') {
	ungetc(c, fp);
	p = buf;
	state = 1;
	break;
      }
      if (c == '/') {
	int c2 = getc(fp);
	if (c2 == '*') {
	  putc(c, stdout);
	  putc(c2, stdout);
	  state = 4;
	  break;
	} else if (c2 == '/') {
	  putc(c, stdout);
	  putc(c2, stdout);
	  do {
	    c = getc(fp);
	    putc(c, stdout);
	  } while(c != '\n');
	  break;
	}
	ungetc(c2, fp);
      }
      if (nl && c == '#') {
	putc(c, stdout);
	do {
	  c = getc(fp);
	  putc(c, stdout);
	} while(c != '\n');
	break;
      }
      putc(c, stdout);
      if (!isspace(c)) nl = false;
      if (c == '\n') nl = true;
      if (c == '\"') state = 2;
      if (c == '\'') state = 3;
      break;

    case 1: // Identifier
      if (isalnum(c) || c == '_') {
	if (p - buf < N) { *p++ = c; *p = '\0'; }
	putc(c, stdout);
      } else if (c == '\"') {
	insert(buf);
	putc(c, stdout);
	state = 2;
      } else if (c == '\'') {
	insert(buf);
	putc(c, stdout);
	state = 3;
      } else {
	insert(buf);
	putc(c, stdout);
	state = 0;
      }
      break;

    case 2: // String
      if (c == '\\') {
	putc(c, stdout);
	putc(getc(fp), stdout);
      } else if (c == '\"') {
	putc(c, stdout);
	state = 0;
      } else {
	putc(c, stdout);
      }
      break;

    case 3: // Character
      if (c == '\\') {
	putc(c, stdout);
	putc(getc(fp), stdout);
      } else if (c == '\'') {
	putc(c, stdout);
	state = 0;
      } else {
	putc(c, stdout);
      }
      break;

    case 4: // Comment
      if (c == '*') {
	int c2 = getc(fp);
	if (c2 == '/') {
	  putc(c, stdout);
	  putc(c2, stdout);
	  state = 0;
	  break;
	}
	ungetc(c2, fp);
      }
      putc(c, stdout);
      break;
    }
  }
}

int main(int argc, char **argv) {
  nalloc = 1;
  keywords = malloc(sizeof(char *) * nalloc);

  if (argc != 4) {
    fprintf(stderr, "%s <input file> <keywords file> <suffix>\n", argv[0]);
    fprintf(stderr, "Add the suffix to keywords\n");
    exit(-1);
  }

  FILE *fp = fopen(argv[2], "r");
  if (fp == NULL) {
    fprintf(stderr, "Cannot open %s\n", argv[2]);
    exit(-1);
  }

  char buf[N];

  while(fgets(buf, N, fp) != NULL) {
    if (strlen(buf) >= 1) buf[strlen(buf)-1] = '\0';
    keywords[nkeywords] = malloc(sizeof(char) * (strlen(buf) + 1));
    strcpy(keywords[nkeywords], buf);
    nkeywords++;
    if (nkeywords >= nalloc) {
      nalloc *= 2;
      keywords = realloc(keywords, sizeof(char *) * nalloc);
    }
  }

  fclose(fp);

  suffix = argv[3];

  fp = fopen(argv[1], "r");
  if (fp == NULL) {
    fprintf(stderr, "Cannot open %s\n", argv[1]);
    exit(-1);
  }

  doit(fp);

  fclose(fp);

  exit(0);
}

// cat sleefinline*.h | egrep -o '[a-zA-Z_][0-9a-zA-Z_]*' | sort | uniq > cand.txt
