#include <gtest/gtest.h>
#include "../src/soa_vector.hpp"

struct Nested
{
    float& n1;
    double& n2;
};

struct TypeRef
{
    float& a;
    int& b;
    Nested nested1;
    Nested nested2;
};

TEST(SOA_VECTOR, IndexingAndReferences)
{
    soa::soa_vector<TypeRef> soa(100);

    GTEST_ASSERT_EQ(soa.size(),100);

    //assign
    soa[1].a=1230;
    soa[1].b=1231;
    soa[1].nested1.n1=1232;
    soa[1].nested2.n1=1233;

    //verify assignment
    GTEST_ASSERT_EQ(soa[1].a,1230);
    GTEST_ASSERT_EQ(soa[1].b,1231);
    GTEST_ASSERT_EQ(soa[1].nested1.n1,1232);
    GTEST_ASSERT_EQ(soa[1].nested2.n1,1233);


    //get reference
    auto ref=soa[2];
    ref.a=1;
    ref.b=2;
    ref.nested1.n1=3.f;
    GTEST_ASSERT_EQ(soa[2].a,1);
    GTEST_ASSERT_EQ(soa[2].b,2);
    GTEST_ASSERT_EQ(soa[2].nested1.n1,3);

    //assigment through reference
    soa[3]=ref;
    GTEST_ASSERT_EQ(soa[3].a,1);
    GTEST_ASSERT_EQ(soa[3].b,2);
    GTEST_ASSERT_EQ(soa[3].nested1.n1,3);

    soa[5]=soa[3];
    GTEST_ASSERT_EQ(soa[5].a,1);
    GTEST_ASSERT_EQ(soa[5].b,2);
    GTEST_ASSERT_EQ(soa[5].nested1.n1,3);
}

TEST(SOA_VECTOR, ElementSnapshot)
{
    soa::soa_vector<TypeRef> vec(10);
    vec[0].a = 1.5f;
    vec[0].b = 2;
    vec[0].nested1.n1 = 3.5f;
    vec[0].nested1.n2 = 4.5;
    vec[0].nested2.n1 = 5.5f;
    vec[0].nested2.n2 = 6.5;

    soa::soa_element_snapshot<TypeRef> snapshot(vec[0]);
    TypeRef reference_to_snapshot = snapshot;

    GTEST_ASSERT_EQ(reference_to_snapshot.a, 1.5f);
    GTEST_ASSERT_EQ(reference_to_snapshot.b, 2);
    GTEST_ASSERT_EQ(reference_to_snapshot.nested1.n1, 3.5f);
    GTEST_ASSERT_EQ(reference_to_snapshot.nested1.n2, 4.5);
    GTEST_ASSERT_EQ(reference_to_snapshot.nested2.n1, 5.5f);
    GTEST_ASSERT_EQ(reference_to_snapshot.nested2.n2, 6.5);

    snapshot = vec[0];
    vec[0].b = 7;
    GTEST_ASSERT_EQ(std::get<1>(snapshot.m_data), 2);

    vec[0] = snapshot;
    GTEST_ASSERT_EQ(vec[0].b, 2);

    GTEST_ASSERT_EQ(reference_to_snapshot.b, 2);
}

TEST(SOA_VECTOR, AllocationCopyAndResize)
{
    soa::soa_vector<TypeRef> vec;
    GTEST_ASSERT_EQ(vec.size(),0);
    GTEST_ASSERT_EQ(vec.capacity(),0);

    vec.reserve(5);
    GTEST_ASSERT_EQ(vec.size(),0);
    GTEST_ASSERT_EQ(vec.capacity(),5);

    vec.resize(100);
    GTEST_ASSERT_EQ(vec.size(),100);
    GTEST_ASSERT_EQ(vec.capacity(),100);

    for (int i=0;i<vec.size();++i)
    {
        vec[i].b=i;
        vec[i].nested1.n2=-i;
    }

    //data preserved after resize
    vec.resize(101);
    GTEST_ASSERT_EQ(vec.size(),101);
    GTEST_ASSERT_EQ(vec.capacity(),101);
    for (int i=0;i<100;++i)
    {
        GTEST_ASSERT_EQ(vec[i].b,i);
        GTEST_ASSERT_EQ(vec[i].nested1.n2,-i);
    }

    //resize down
    vec.resize(50);
    vec.shrink_to_fit();
    GTEST_ASSERT_EQ(vec.size(),50);
    GTEST_ASSERT_EQ(vec.capacity(),50);
    for (int i=0;i<vec.size();++i)
    {
        GTEST_ASSERT_EQ(vec[i].b,i);
        GTEST_ASSERT_EQ(vec[i].nested1.n2,-i);
    }

    //allocate range
    auto start=vec.allocate(1000);
    GTEST_ASSERT_EQ(vec.size(),1050);
    GTEST_ASSERT_GE(vec.capacity(),1050);
    for (int i=0;i<50;++i)
    {
        GTEST_ASSERT_EQ(vec[i].b,i);
        GTEST_ASSERT_EQ(vec[i].nested1.n2,-i);
    }

    //copy constructor
    auto vec_copy=vec;
    GTEST_ASSERT_EQ(vec_copy.size(),vec.size());
    GTEST_ASSERT_EQ(vec_copy.capacity(),vec.size());
    for (int i=0;i<50;++i)
    {
        GTEST_ASSERT_EQ(vec_copy[i].b,i);
        GTEST_ASSERT_EQ(vec_copy[i].nested1.n2,-i);
    }

    //clear
    auto vec_moved=std::move(vec);
    GTEST_ASSERT_EQ(vec.size(),0);
    GTEST_ASSERT_EQ(vec.capacity(),0);

    GTEST_ASSERT_EQ(vec_moved.size(),1050);
    for (int i=0;i<50;++i)
    {
        GTEST_ASSERT_EQ(vec_moved[i].b,i);
        GTEST_ASSERT_EQ(vec_moved[i].nested1.n2,-i);
    }

    vec_moved.clear();
    GTEST_ASSERT_EQ(vec.size(),0);
    GTEST_ASSERT_EQ(vec.capacity(),0);
}

TEST(SOA_VECTOR,IteratorsAndStd)
{
    soa::soa_vector<TypeRef> vec(250);
    int i=0;
    for (auto&& v:vec)//foreach
    {
        v.b = i;//init .b in ascending order
        v.nested2.n2=2*i;
        ++i;
    }

    //verify data through index reference
    for (i=0;i<vec.size();++i)
    {
        GTEST_ASSERT_EQ(vec[i].b,i);
        GTEST_ASSERT_EQ(vec[i].nested2.n2, 2*i );
    }

    //sort in descending by .b
    std::sort(vec.begin(),vec.end(),[](TypeRef l, TypeRef r){return l.b>r.b;});

    //verify sorting
    for (i=0;i<vec.size();++i)
    {
        auto j=vec.size()-i-1;//reversed index
        GTEST_ASSERT_GE(vec[j].b,i);
        GTEST_ASSERT_EQ(vec[j].nested2.n2, 2*i );//key-value relation preserved
    }
}
