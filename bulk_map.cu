#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/merge.h>
#include <thrust/unique.h>
#include <thrust/set_operations.h>
#include <thrust/inner_product.h>
#include <thrust/detail/internal_functional.h> // for thrust::detail::not2

// TODO consider bulk_unordered_map with a hash(key) for faster duplicate testing?
// TODO keep a smaller auxillary bulk_map around until its large enough for merge (or forced to merge)?
// TODO consider non-member functions (e.g. union(A,B,C), intersection(A,B,C))

// XXX provide insert_intersection method?
// XXX why is map<K,V>::value_type pair<const K, V>
// XXX Note: "value" is interpreted as pair<Key,T> in the STL, rather than just T

template <typename Compare>
struct compare_tuple0
{
  Compare comp;

  compare_tuple0(Compare comp)
    : comp(comp)
  {}

  template <typename Tuple1, typename Tuple2>
  __host__ __device__
  bool operator()(const Tuple1& a, const Tuple2& b) const
  {
    return comp(thrust::get<0>(a), thrust::get<0>(b));
  }
};

template <typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2,
          typename StrictWeakCompare>
thrust::pair<OutputIterator1,OutputIterator2>
  merge_by_key(InputIterator1 first1, InputIterator1 last1,
               InputIterator2 first2,
               InputIterator3 first3, InputIterator3 last3,
               InputIterator4 first4,
               OutputIterator1 output1,
               OutputIterator2 output2,
               StrictWeakCompare comp)
{
  thrust::tuple<OutputIterator1,OutputIterator2> output =
    thrust::merge
      (thrust::make_zip_iterator(thrust::make_tuple(first1, first2)),
       thrust::make_zip_iterator(thrust::make_tuple(last1,  first2)),
       thrust::make_zip_iterator(thrust::make_tuple(first3, first4)),
       thrust::make_zip_iterator(thrust::make_tuple(last3,  first4)),
       thrust::make_zip_iterator(thrust::make_tuple(output1, output2)),
       compare_tuple0<StrictWeakCompare>(comp)).get_iterator_tuple();

  return thrust::make_pair(thrust::get<0>(output), thrust::get<1>(output));
}

template <typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2,
          typename StrictWeakCompare>
thrust::pair<OutputIterator1,OutputIterator2>
  set_union_by_key(InputIterator1 first1, InputIterator1 last1,
                   InputIterator2 first2,
                   InputIterator3 first3, InputIterator3 last3,
                   InputIterator4 first4,
                   OutputIterator1 output1,
                   OutputIterator2 output2,
                   StrictWeakCompare comp)
{
  thrust::tuple<OutputIterator1,OutputIterator2> output =
    thrust::set_union
      (thrust::make_zip_iterator(thrust::make_tuple(first1, first2)),
       thrust::make_zip_iterator(thrust::make_tuple(last1,  first2)),
       thrust::make_zip_iterator(thrust::make_tuple(first3, first4)),
       thrust::make_zip_iterator(thrust::make_tuple(last3,  first4)),
       thrust::make_zip_iterator(thrust::make_tuple(output1, output2)),
       compare_tuple0<StrictWeakCompare>(comp)).get_iterator_tuple();

  return thrust::make_pair(thrust::get<0>(output), thrust::get<1>(output));
}

template <typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2,
          typename StrictWeakCompare>
thrust::pair<OutputIterator1,OutputIterator2>
  set_difference_by_key(InputIterator1 first1, InputIterator1 last1,
                        InputIterator2 first2,
                        InputIterator3 first3, InputIterator3 last3,
                        InputIterator4 first4,
                        OutputIterator1 output1,
                        OutputIterator2 output2,
                        StrictWeakCompare comp)
{
  thrust::tuple<OutputIterator1,OutputIterator2> output =
    thrust::set_difference
      (thrust::make_zip_iterator(thrust::make_tuple(first1, first2)),
       thrust::make_zip_iterator(thrust::make_tuple(last1,  first2)),
       thrust::make_zip_iterator(thrust::make_tuple(first3, first4)),
       thrust::make_zip_iterator(thrust::make_tuple(last3,  first4)),
       thrust::make_zip_iterator(thrust::make_tuple(output1, output2)),
       compare_tuple0<StrictWeakCompare>(comp)).get_iterator_tuple();

  return thrust::make_pair(thrust::get<0>(output), thrust::get<1>(output));
}


template <typename InputIterator,
          typename BinaryPredicate>
size_t unique_count(InputIterator first, InputIterator last, BinaryPredicate pred)
{
  // XXX violates InputIterator semantics
  if (first == last)
    return 0;
  else
    return thrust::inner_product(first, last - 1,
                                 first + 1,
                                 size_t(1),
                                 thrust::plus<size_t>(),
                                 thrust::detail::not2(pred));
}


template <typename KeyContainer,
          typename ValueContainer,  // MappedContainer? TContainer? ValueContainer?
          typename KeyCompare = thrust::less<typename KeyContainer::value_type> >
class bulk_map
{
  public:
    typedef KeyContainer    key_container_type; // like std::priority_queue
    typedef ValueContainer  value_container_type;

    // XXX should we expose these?
    // these types have no STL analog, but are similar to Python dictionary iterators (i.e. dict.keys() dict.values())
    typedef typename key_container_type::iterator         key_iterator;
    typedef typename value_container_type::iterator       value_iterator;
    typedef typename key_container_type::const_iterator   const_key_iterator;
    typedef typename value_container_type::const_iterator const_value_iterator;

    // XXX make these return pairs instead w/ transform_iterator<zip_iterator<...>, ... > ?
    typedef thrust::zip_iterator<thrust::tuple<key_iterator, value_iterator> >             iterator;
    typedef thrust::zip_iterator<thrust::tuple<const_key_iterator, const_value_iterator> > const_iterator;

    typedef KeyCompare key_compare;  // like std::map

    typedef typename KeyContainer::value_type   key_type;    // like std::map or std::unordered_map
    typedef typename ValueContainer::value_type mapped_type; // like std::map or std::unordered_map

    // XXX is this the right value type?
    //typedef thrust::pair<const key_type, mapped_type> value_type; // like std::map or std::unordered_map
    typedef thrust::tuple<key_type, mapped_type> value_type;

    // TODO reverse_iterator, const_reverse_iterator
    // TODO value_compare

  protected:
    KeyContainer   m_keys;
    ValueContainer m_vals;
    KeyCompare     m_comp;

  void normalize(void)
  {
    // sort pairs and remove duplicates
    thrust::sort_by_key(m_keys.begin(), m_keys.end(), m_vals.begin(), m_comp);

    m_keys.resize(thrust::unique_by_key(m_keys.begin(), m_keys.end(), m_vals.begin(), thrust::detail::not2(m_comp)).first - m_keys.begin());
  }

  void check_sizes(void) const
  {
    if (m_keys.size() != m_vals.size())
      throw std::invalid_argument("key and value vectors must have same size");
  }

  public:
    bulk_map(void)
    {}
  
    // TODO enable sorting bypass w/ special tag? or provide bulk_map_view<KeyRange,ValueRange, .. >
    bulk_map(const KeyContainer& keys,
             const ValueContainer& values,
             const KeyCompare& comp = KeyCompare())
      : m_keys(keys), m_vals(values), m_comp(comp)
    {
      check_sizes();
      normalize();
    }
    
    template <typename InputIterator1, typename InputIterator2>
    bulk_map(InputIterator1 first1, InputIterator1 last1,
             InputIterator2 first2, InputIterator2 last2,
             const KeyCompare& comp = KeyCompare())
      : m_keys(first1, last1), m_vals(first2, last2), m_comp(comp)
    {
      check_sizes();
      normalize();
    }
  
    // TODO compare src->temp/swap approach against copy/temp->this approach
    void insert(const bulk_map& other)
    {
      // XXX optimize when *this < other or other < *this or this == &other
      KeyContainer   temp_keys(size() + other.size()); // XXX fill value? ideally we'd use uninitialized storage w/ special iterator adaptor
      ValueContainer temp_vals(size() + other.size());

      // note: other's keys must come first
      temp_keys.resize
        (set_union_by_key(other.m_keys.begin(), other.m_keys.end(),
                          other.m_vals.begin(),
                          m_keys.begin(), m_keys.end(),
                          m_vals.begin(),
                          temp_keys.begin(),
                          temp_vals.begin(),
                          m_comp).first - temp_keys.begin());
      temp_vals.resize(temp_keys.size());

      m_keys.swap(temp_keys);
      m_vals.swap(temp_vals);
    }

    void erase(const bulk_map& other)
    {
      // XXX optimize when *this < other or other < *this or this == &other
      KeyContainer   temp_keys(m_keys);
      ValueContainer temp_vals(m_vals);

      m_keys.resize
        (set_difference_by_key(temp_keys.begin(), temp_keys.end(),
                               temp_vals.begin(),
                               other.m_keys.begin(), other.m_keys.end(),
                               other.m_vals.begin(),
                               m_keys.begin(),
                               m_vals.begin(),
                               m_comp).first - m_keys.begin());
      m_vals.resize(m_keys.size());
    }

    // XXX is combine the best word? want something that implies two-ness instead of 1 + many
    template <typename BinaryFunction>
    void combine(const bulk_map& other, BinaryFunction binary_op)
    {
      // XXX optimize when *this < other or other < *this or this == &other
      KeyContainer   temp_keys(size() + other.size()); // XXX fill value? ideally we'd use uninitialized storage w/ special iterator adaptor
      ValueContainer temp_vals(size() + other.size());

      // note: other's keys must come first
      merge_by_key(other.m_keys.begin(), other.m_keys.end(),
                   other.m_vals.begin(),
                   m_keys.begin(), m_keys.end(),
                   m_vals.begin(),
                   temp_keys.begin(),
                   temp_vals.begin(),
                   m_comp);

      size_t num_unique = unique_count(temp_keys.begin(), temp_keys.end(), thrust::detail::not2(m_comp));

      m_keys.resize(num_unique);
      m_vals.resize(num_unique);

      thrust::reduce_by_key(temp_keys.begin(), temp_keys.end(),
                            temp_vals.begin(),
                            m_keys.begin(),
                            m_vals.begin(),
                            thrust::detail::not2(m_comp),
                            binary_op);
    }

    size_t size(void) const
    {
      return m_keys.size();
    }

    void shrink_to_fit(void)
    {
      m_keys.shrink_to_fit();
      m_vals.shrink_to_fit();
    }

    iterator begin(void)
    {
      return iterator(thrust::make_tuple(m_keys.begin(), m_vals.begin()));
    }
    
    iterator cbegin(void) const
    {
      return const_iterator(thrust::make_tuple(m_keys.cbegin(), m_vals.cbegin()));
    }

    iterator begin(void) const
    {
      return cbegin();
    }

    iterator end(void)
    {
      return iterator(thrust::make_tuple(m_keys.end(), m_vals.end()));
    }
    
    iterator cend(void) const
    {
      return const_iterator(thrust::make_tuple(m_keys.cend(), m_vals.cend()));
    }

    iterator end(void) const
    {
      return cend();
    }
};

template <typename BulkMap>
void print(std::string s, BulkMap& m)
{
  std::cout << s << " :";
  for (typename BulkMap::const_iterator iter = m.begin(); iter != m.end(); ++iter)
  {
    typedef typename BulkMap::value_type T;
    T x = *iter;
    std::cout << " (" << thrust::get<0>(x) << "," << thrust::get<1>(x) << ")";
  }
  std::cout << std::endl;
}

int main(void)
{
  int K1[] = {3,0,9,3,8,2,4,0,9,8};
  int V1[] = {1,1,1,1,1,1,1,1,1,1};
  size_t N1 = sizeof(K1) / sizeof(*K1);

  int K2[] = {5,3,0,6,2,1,4,7,2};
  int V2[] = {2,2,2,2,2,2,2,2,2};
  size_t N2 = sizeof(K2) / sizeof(*K2);

  typedef thrust::device_vector<int> KeyContainer;
  typedef thrust::device_vector<int> ValueContainer;

  typedef bulk_map<KeyContainer,ValueContainer> BulkMap;

  thrust::plus<int> Op;
  
  BulkMap A(K1, K1 + N1, V1, V1 + N1);
  BulkMap B(K2, K2 + N2, V2, V2 + N2);

  print("A",A);
  print("B",B);
  
  { BulkMap C(A); C.insert(B);     print("A.insert(B)   ",C); }
  { BulkMap C(A); C.erase(B);      print("A.erase(B)    ",C); }
  { BulkMap C(A); C.combine(B,Op); print("A.combine(B,+)",C); }

  return 0;
}

