#include <random>
#include <complex>
#include <vector>
#include <iostream>

using namespace std;

const size_t V=64*64*64*128*12;
//const size_t V=64;

using Compl=
  complex<double>;

template <typename T>
struct Vect :
  vector<complex<T>>
{
  const string t;
  
  Vect(const size_t& v,
       const string& t,
       mt19937_64& gen) :
    vector<complex<T>>(v),
    t(t)
  {
    fill(gen);
  }
  
  void fill(mt19937_64& gen)
  {
    for(auto& c : *this)
	c=Compl(std::normal_distribution<>()(gen),std::normal_distribution<>()(gen));
    cout<<"Vector "<<t<<" filled"<<endl;
  }
  
  complex<T> getProjectionWith(const Vect& oth) const
  {
    complex<T> s=0;
    
    for(size_t i=0;i<this->size();i++)
      s+=conj(oth[i])*(*this)[i];
    
    return s;
  }
  
  T norm2() const
  {
    T s=0;
    
    for(size_t i=0;i<this->size();i++)
      s+=norm((*this)[i]);
    
    return s;
  }
  
  void normalize()
  {
    const T aa=
      norm2();
    cout<<"|"<<t<<"|^2: "<<aa<<endl;
    
    const T a=
      sqrt(aa);
    
    for(auto& c : *this)
      c/=a;
    
    cout<<t<<" normalized"<<endl;
  }
  
  void projectWith(const Vect& oth)
  {
    const complex<T> p=
      getProjectionWith(oth);
    cout<<t<<" projection on "<<oth.t<<": "<<p<<endl;
    
    for(size_t i=0;i<this->size();i++)
      (*this)[i]-=p*oth[i];
    cout<<t<<" projected away from "<<oth.t<<endl;
  }
  
  void truncateToFloat()
  {
    for(auto& c : *this)
      c=complex<float>(c);
    cout<<t<<" truncated"<<endl;
  }
};


int main()
{
  mt19937_64 gen(235235523);
  cout<<"Volume: "<<V<<endl;
  
  Vect<double> a(V,"a",gen);
  
  Vect<double> b(V,"b",gen);
  
  a.normalize();
  
  b.projectWith(a);
  
  b.normalize();
  
  const Compl pp=
    a.getProjectionWith(b);
  cout<<"Verify projection: "<<pp<<endl;
  
  for(auto & v : {&a,&b})
    v->truncateToFloat();
  
  const Compl ppf=
    a.getProjectionWith(b);
  cout<<"Projection after cast to float: "<<ppf<<endl;
  
  return 0;
}
