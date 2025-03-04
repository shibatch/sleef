//   Copyright Naoki Shibata and contributors 2010 - 2025.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <cstdio>
#include <vector>
#include <unordered_map>

using namespace std;

template <>
struct std::hash<vector<char>> {
  size_t operator()(const vector<char> &v) const {
    size_t u = 0;
    for(size_t i=0;i<v.size();i++) {
      int b = (v.data()[i] + i) & ((sizeof(u)*8)-1);
      u = (u << b) | (u >> ((sizeof(u)*8)-b));
      u += v.data()[i] + i;
    }
    return u;
  }
};

class Serializer {
public:
  virtual void write(const void *, size_t) = 0;
  virtual void flush() {}
};

class Deserializer {
public:
  virtual void read(void *, size_t) = 0;

  template<typename T, typename enable_if<(is_trivially_copyable<T>::value), int>::type = 0>
  T read() {
    T t;
    read(&t, sizeof(T));
    return t;
  }
};

class FileSerializer : public Serializer {
  FILE *fp;

public:
  FileSerializer(FILE *fp_) : fp(fp_) {}

  void write(const void *p, size_t z) {
    fwrite(p, z, 1, fp);
  }

  void flush() { fflush(fp); }
};

class FileDeserializer : public Deserializer {
  FILE *fp;

public:
  FileDeserializer(FILE *fp_) : fp(fp_) {}

  void read(void *p, size_t z) {
    if (!fread(p, z, 1, fp)) throw(runtime_error("FileDeserializer::read : could not read"));
  }
};

template<typename T, typename enable_if<(is_trivially_copyable<T>::value), int>::type = 0>
Serializer& operator<<(Serializer &s, const T& v) {
  s.write((const char *)&v, sizeof(v));
  return s;
}

template<typename T, typename enable_if<(is_trivially_copyable<T>::value), int>::type = 0>
Deserializer& operator>>(Deserializer &s, T& v) {
  s.read((char *)&v, sizeof(v));
  return s;
}

template<typename T>
Serializer& operator<<(Serializer &s, const vector<T>& v) {
  s << v.size();
  for(size_t i=0;i<v.size();i++) s << v.data()[i];
  return s;
}

template<typename T>
Deserializer& operator>>(Deserializer &d, vector<T>& v) {
  size_t z = d.read<size_t>();
  for(size_t i=0;i<z;i++) {
    T t;
    d >> t;
    v.push_back(t);
  }
  return d;
}

Serializer& operator<<(Serializer &s, const string& str) {
  s << (str.size() + 1);
  s.write(str.c_str(), str.size() + 1);
  return s;
}

Deserializer& operator>>(Deserializer &d, string& str) {
  vector<char> v;
  d >> v;
  str = v.data();
  return d;
}

template<typename KT, typename VT>
Serializer& operator<<(Serializer &s, const unordered_map<KT, VT>& m) {
  s << m.size();
  for(auto a : m) s << a.first << a.second;
  return s;
}

template<typename KT, typename VT>
Deserializer& operator>>(Deserializer &d, unordered_map<KT, VT>& m) {
  size_t z = d.read<size_t>();
  for(size_t i=0;i<z;i++) {
    KT key;
    d >> key;
    VT value;
    d >> value;
    m[key] = value;
  }
  return d;
}
