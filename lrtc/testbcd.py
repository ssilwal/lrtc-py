import bcd

def test_compute_x():
   ans = bcd.compute_x([1,2,3,4], [1,2,3,4,4])
   print(ans)
   assert(ans == 10)

def main():
   test_compute_x()

if (__name__ == "__main__"):
   main()
