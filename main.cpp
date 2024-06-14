#include <omp.h>
#include <random>
#include <complex>
#include <vector>
#include <iostream>

using namespace std;

const size_t V=64*64*64*128*12;

using Compl=
  complex<double>;

#pragma omp declare			\
  reduction(+ :				\
	    std::complex<double> :	\
	    omp_out += omp_in )		\
  initializer(omp_priv = omp_orig)

double truncateToNbits(const double& x,
		       const size_t& N)
{
  const double t=
    (1lu<<(53-N))*x;
  
  return t-(t-x);
}

struct Vect :
  vector<complex<double>>
{
  const string t;
  
  Vect(const size_t& v,
       const string& t) :
    vector<complex<double>>(v),
    t(t)
  {
  }
  
  void fill(mt19937_64& gen)
  {
    for(auto& c : *this)
	c=Compl(std::normal_distribution<>()(gen),std::normal_distribution<>()(gen));
    cout<<"Vector "<<t<<" filled"<<endl;
  }
  
  complex<double> getProjectionWith(const Vect& oth) const
  {
    complex<double> s=0;
    
#pragma omp parallel for reduction(+:s)
    for(size_t i=0;i<this->size();i++)
      s+=conj(oth[i])*(*this)[i];
    
    return s;
  }
  
  double norm2() const
  {
    double s=0;
    
#pragma omp parallel for reduction(+:s)
    for(size_t i=0;i<this->size();i++)
      s+=norm((*this)[i]);
    
    return s;
  }
  
  void normalize()
  {
    const double aa=
      norm2();
    cout<<"|"<<t<<"|^2: "<<aa<<endl;
    
    const double a=
      sqrt(aa);
    
#pragma omp parallel for
    for(auto& c : *this)
      c/=a;
    
    cout<<t<<" normalized, check |"<<t<<"|^2-1: "<<norm2()-1<<endl;
  }
  
  void projectWith(const Vect& oth)
  {
    const complex<double> p=
      getProjectionWith(oth);
    cout<<t<<" projection on "<<oth.t<<": "<<p<<endl;
    
#pragma omp parallel for
    for(size_t i=0;i<this->size();i++)
      (*this)[i]-=p*oth[i];
    cout<<t<<" projected away from "<<oth.t<<", residual projection: "<<getProjectionWith(oth)<<endl;
  }
  
  Vect truncateToFloat() const
  {
    Vect out(this->size(),t+"f");
    
#pragma omp parallel for
    for(size_t i=0;i<this->size();i++)
      out[i]=complex<float>((*this)[i]);
    
    cout<<t<<" truncated to float"<<endl;
    
    return out;
  }
  
  Vect truncateToNbits(const size_t& N) const
  {
    Vect out(this->size(),t+"f");
    
#pragma omp parallel for
    for(size_t i=0;i<this->size();i++)
      out[i]=complex<double>(::truncateToNbits((*this)[i].real(),N),
			     ::truncateToNbits((*this)[i].imag(),N));
    
    return out;
  }
  
  Vect(const Vect&)=default;
};


int main()
{
  cout<<"Going to work with "<<omp_get_max_threads()<<" threads"<<endl;
  cout.precision(16);
  // cout<<M_PI<<" "<<endl;
  // cout<<(float)M_PI<<endl;
  // cout<<truncateToNbits(M_PI,24)<<endl;
  
  mt19937_64 gen(235235523);
  cout<<"Volume: "<<V<<endl;
  
  Vect a(V,"a");
  a.fill(gen);
  
  Vect b(V,"b");
  b.fill(gen);
  
  a.normalize();
  
  b.projectWith(a);
  
  b.normalize();
  
  const Compl pp=
    a.getProjectionWith(b);
  cout<<"Verify projection: "<<pp<<endl;
  
  const Vect af=
    a.truncateToFloat();
  
  const Vect bf=
    b.truncateToFloat();
  
  const Compl ppf=
    af.getProjectionWith(bf);
  cout<<"Projection after cast to float: "<<ppf<<endl;
  
  for(size_t nBits=24;nBits>=10;nBits--)
    {
      const Vect at=
	a.truncateToNbits(nBits);
      
      const Vect bt=
	b.truncateToNbits(nBits);
      
      const Compl ppt=
	at.getProjectionWith(bt);
      
      cout<<"Projection after cast to "<<nBits<<" bits: "<<ppt<<endl;
    }
  
  return 0;
}
