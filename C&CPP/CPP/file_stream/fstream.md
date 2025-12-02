### 輸入流與輸出流
在 C++ 裡面，只要跟輸入輸出有關，就會產生一個流，就是常說的 stream，還記得現在學的最基本的兩個 stream 就是 cin 與 cout，我們總是在我們的程式第一行寫上 #include <iostream>，這樣我們的程式才會認得 cin 跟 cout。

全名是 file stream，也就是檔案的 stream，所以他對輸入與輸出，也會有像 cin 跟 cout 的東西，而針對檔案資料的讀入我們叫做 ifstream (input file stream，從檔案讀進來)，輸出叫做ofStream (output file stream，從程式輸出進檔案)。
他們的功用跟 cin 還有 cout 一模一樣，但就只是針對檔案用的。

<p><span style="color:#ff0000; font-size:20px; font-weight:bold;">要使用file stream的話，需要引入的標頭檔：&lt;fstream&gt;</span></p>

``` cpp
//一般的輸入跟輸出
cin >> a >> b;
cout << a << b;
//檔案的輸入跟輸出，不是這樣，這樣是錯誤的!!!!
ifstream >> a >> b;
ofstream >> a >> b;
```

ifstream 跟 ofstream 是類別，類別需要實體化一個物件才能拿來用。

```cpp
#include <iostream>
#include <fstream>
using namespace std;
int main(){
    ifstream in;
    ofstream out;
    string a,b; // 假設檔案內的資料是字串
    in >> a >> b; // 把檔案的資料讀進來給 a 跟 b 兩個變數
    out << "hello, i'm orange"; // 把這句話輸出至一個檔案裡面
}
```

<span style="color:#4A4AFF; font-size:20px; font-weight:bold; background-color:#F0F0F0F0;">in 就是針對你接下來在程式裡面所有跟檔案讀入有關的內容的那個物件的名字，而 out 就是針對檔案輸出。</span>


### 檔案開啟
檔案開啟的語法很簡單，首先的思維是我們一定會有一個要被開啟的檔案。語法如下。記得 in 做檔案資料的讀入，out 做把資料寫入進檔案的動作。

- in.open("data.txt");
    這樣就是告訴程式說，我開啟一個名為 data 的檔案，接下來要從這個文字檔讀資料進來。

- out.open("output.txt");
    這樣是說，我開啟一個名為 output 的檔案，我之後要把資料寫進去這個檔案裡。

### 關閉串流
而你的程式在對檔案的操作結束之後，要把串流關閉，養成好習慣，讓他釋放記憶體。

- in.close()
    關閉資料讀入串流
- out.close()
    關閉資料輸出串流
### 檔案開啟失敗
如果今天你路徑設置錯誤，檔名打錯，程式是有可能找不到檔案的，針對這件事 C++ 也有處理的方式喔。他提供了一個名為 fail 的 function 用來確認檔案是否成功開啟。
in.fail()，這個函式用來確認檔案是否成功開啟。如果檔案開啟失敗，會回傳 1，檔案成功開啟，會回傳 0。

### 檔案開啟小總結
針對上面的內容，我們來看一隻範例程式。

``` cpp
#include <iostream>
#include <fstream>
#include <cstdlib>    //使用exit必須include
using namespace std;
int main(){
    ifstream in;
    ofstream out;
    //假設檔案的名字為 data.txt，但我們這邊打錯字成 dat.txt
    in.open("dat.txt");
    if(in.fail()){ // 我們應該會進來這個條件判斷裡面，因為找不到 dat.txt 這個檔案
        cout << "input file opening failed";
        exit(1); // 程式錯誤終止
    }
    out.open("output.txt");
    if(out.fail()){
        cout << "output file opening failed";
        exit(1);
    }
    in.close();
    out.close();
    return 0;
}
```

這邊特別談論一下 exit 這個 function。這個函式的功能是終止程式，如果裡面參數放 0 或是空的，代表程式正常終止，但如果放其他數字，表示程式非正常終止。
今天在這邊的情況，條件判斷檔案沒有正常開啟，所以我們認為程式應該要出錯，所以我們裡面參數放 1 表示程式出錯，錯誤終止。
而 exit 這個函式是隸屬在 cstdlib 這個函式庫裡面，所以記得 incldue 進來。

### 資料讀入
接著來談從檔案資料讀入資料進來，假設有一個名為 data.txt 的文字檔，裡面有 3 個數字分別是 1、2、3，我們想透過程式把他們讀進來。

``` cpp
int main(){
    ifstream in;
    in.open("data.txt");
    //我忽略 in.fail() 檢查檔案開啟的步驟哦
    int fir, sec, thi;
    in >> fir >> sec >> thi; // fir = 1, sec = 2, thi = 3
    cout << fir+sec+thi; // 印出 6
    return 0;
}
```

透過上面的語法，你應該有感覺，讀過一個資料(每 in 一次，上面雙箭頭三個，所以 in 三次)後就會往後再讀下一個，而我們讀入三次存給三個變數，之後輸出結果。
那我想問題很明顯，今天如果有 100 個數字，難道要 in 雙箭頭 100 次?

所以你已知有三個數字，你可能會這樣寫。

``` cpp
int main(){
    ifstream in;
    in.open("data.txt");
    int sum = 0, value = 0;
    for(int i = 0; i < 3; i++){
        in >> value;
        sum += value;
    }
    cout << sum;
    return 0;
}
```

好，問題又來了，如果今天一份檔案，你不知道有幾個數字，那怎麼辦呢?我又不知道 for 迴圈要讓他跑幾次，我也不知道什麼時候檔案會被我讀到結束，所以 C++ 針對這個問題有下面兩個處理方式(其實不只，這邊只舉兩個)。

- 第一種方法 - 相同類型連續讀取

``` cpp
int main(){
    ifstream in;
    in.open("data.txt");
    int sum = 0, value = 0;
    while(in >> value){ //只要還沒讀到完，條件成立就繼續一直讀
        sum += value;
    }
    cout << sum;
    return 0;
}
```

但這樣寫會有什麼問題?今天你檔案內的資料不會這麼剛好的全部都是數字吧!不可能這麼剛好在第五行的 in 讀入都是數字，而且你 value 變數的型態是 int，那如果一份檔案同時需要讀入字串怎麼辦?

- 第二種方法 - eof

``` cpp
int main(){
    ifstream in;
    in.open("data.txt");
    int sum = 0, value = 0;
    while(!in.eof()){ //只要還沒讀到完，條件成立就繼續一直讀
        in >> value;
        sum += value;
    }
    cout << sum;
    return 0;
}
```

這邊要來談 in.eof()這個函式，全名是 end of file，用來判斷是不是讀取到資料尾巴，這個函式會有一個指針，一個一個的去遍歷你的檔案，假設今天你檔案內的資料長下面這樣。假設 6 後面是 沒有 空格跟換行符號之類的字符。

in.eof()並不會去檔案裡讀取任何資料, 它只是回報上一次由檔案串流讀取資料時的狀態。

如何處理最後的那個空格所帶來的錯誤
我們先看 Code! 這樣的程式就算你檔案最後有空格，也還是會印出正解 21。

``` cpp
int main(int argc, char** argv) {
    ifstream in;
    ofstream out;
    in.open("data.txt");
    if(in.fail()){
    	cout << "input file opening failed\n";
    	exit(1);
    }
    int sum = 0, value = 0;
    int a, b;
    while(!in.eof()){
    	in >> value;
    	if (!in.fail()){
    	    cout << value << endl;
    	    sum += value;
    	}
    }
    cout << sum;
    return 0;
}
```

### 隱藏的 flag

``` cpp
#include <iostream>
#include <fstream>
using namespace std;
int main(){
    //假設檔案內有 1~6 6個整數，假設檔案的最後有一個空格字符
    ifstream in;
    in.open("data.txt");
    if(in.fail()){
        cout << "file opening is failed...";
        exit(1);
    }
    while (!in.eof())
    {
        in >> value;
        cout << value << endl;
        cout << "good()=" << in.good() << ",";
        cout << "fail()=" << in.fail() << ",";
        cout << "bad()=" << in.bad() << ",";
        cout << "eof()=" << in.eof() <<endl;
    }
    return 0;
}
```
![[隱藏的flag.png]]

在輸入輸出裡面其實針對大部分的操作背後是有 state flag 確認你執行的情況的，而每一次的讀寫操作或函式呼叫 (包括 in.eof()、輸入(>>)、輸出(<<)) 都可以在他們的下一步執行四種函式，去確認我們上一次針對串流所做的操作而得到的狀態。 (也就是我們這裡的 in，平常就是我們的 cin 跟 cout)
以第 7 行開檔案為例，在呼叫 open 函式之後第 8 行我們呼叫 fail 函式，來確認檔案是不是正確開啟，這邊就是先針對 in 做了函式呼叫，接著進行狀態確認。
而如果檔案正確開啟，甚麼事都沒發生，上一次的串流操作正確，所以狀態就是 good，這個時候呼叫 in.good() 會得到 1(true)，呼叫 in.fail() 會得到 0(false)。
為了確認到這邊你有懂，請問第 16 到第 19 行這四個狀態函式是去確認哪一行的串流操作呢?
第 14 行的資料讀入哦
因為他是 16 行呼叫函式前 上一次 的讀入串流操作

這個時候我們來看看上面的輸出吧!每次進入迴圈第 14 行都會讀入一個數字，而前 6 次 (包括第六次) 都能正確地讀到一個數字並傳給 value 變數，所以印出四個狀態來檢視，都會是 good 為 true。其他三個狀態為 false。

每一次讀入數字後，會去看有沒有下一個數字，並把指針停留在下一個數字的位置。

所以第一次的讀入讀到 1 接著讀發現有下一個數字，指針停在這(數字 2)，再進第二次迴圈，把 2 給 value 變數，指針停在 3，一直到第六次迴圈，把 6 給 value 變數，但他發現後面只剩下一個空格但沒有數字了，所以指針停在 6 後面的空格上，但還沒有讀到檔案結束字符 0xFF。

所以第七次迴圈進來，空格無法指派給 value，而 value 的值是第六次的 6，所以最後 6 多印一次。

而下面的四個狀態函式在檢視第七次的 in >> value 的時候，讀入是失敗的，因為空格無法 assign，所以你會發現 fail() 回傳 true，而在第七次的時候空格讀完往下就看到了結束字符，所以 eof() 也回傳 true。

那你應該有發現上面的輸出在第七次的時候印出的是 fail 為 true，eof 也為 true。

- fail 為 true
    - 代表發生錯誤，那請問第七次在 in >> value 發生甚麼錯呢?
      - 第七次讀入的是空格，但我們要求寫入進 value，兩者型態不一致，無法寫入，所以發生 fail，fail 變成 true。
- eof (end of file) 為 true
  - eof 為 true 表示檔案真的讀到結尾了 (讀取到結束字符 0xFF)
  
總結來說，我們有四個檢查狀態的函式，至於他們的效用，你可以點超連結進去看。

- [good()](http://www.cplusplus.com/reference/ios/ios/good/)
- [fail()](http://www.cplusplus.com/reference/ios/ios/fail/)
- [bad()](http://www.cplusplus.com/reference/ios/ios/bad/)
- [eof()](http://www.cplusplus.com/reference/ios/ios/eof/)

最後來看上面處理空格錯誤的那個例子，為什麼這樣做就能得到 21 呢?我把上面的 code 複製下來。

``` cpp
int main(int argc, char** argv) {
    ifstream in;
    ofstream out;
    in.open("data.txt");
    if(in.fail()){
    	cout << "input file opening failed\n";
    	exit(1);
    }
    int sum = 0, value = 0;
    int a, b;
    while(!in.eof()){
    	in >> value;
    	if (!in.fail()){
    	    cout << value << endl;
    	    sum += value;
    	}
    }
    cout << sum;
    return 0;
}
```

現在每一次的讀入都使用 fail 函式來檢查狀態，第七次因為是 fail 為 true，所以條件判斷失敗，第七次的迴圈不會進第 14 行，所以最後印出 21。

### 資料寫出
有寫入自然就有寫出，透過程式將文字輸出到檔案是非常正常的事情，比方說一些 log 檔案(記錄檔)，用來記錄一些電腦的狀態，使用者的操作等等，將這些內容寫到文字檔內，能夠有效的紀錄大家的操作，後續要追蹤你以前的操作的時候，就可以透過閱讀你存下來的這些文字檔來做到這件事。

資料寫出檔案開啟
其實跟輸入大同小異，語法上完全一樣。小細節的不一樣你應該很直覺了，就不再細提(比方說箭頭方向)。而我們現在有一個什麼內容都沒有的檔案 input.txt。

在程式內，將他打開之後才能夠做寫入。

``` cpp
int main(){
    ofstream out; // 建立輸出串流物件
    out.open("input.txt"); // 開啟即將被寫入資料的檔案
    if(out.fail()){ //確認是否開啟成功
        cout << "input file opening failed...";
        exit(1);
    }
    string a = "hello, i'm Orange", b = "Yin-Ho is very handsome!";
    out << a << endl << b; // 將兩個字串寫入
}
```

所以你一樣可以迴圈，然後把想輸入的東西陸續輸入，就完全依照你的情況去設計程式碼。

而因為是串流，所以他也有四個狀態函式喔，雖然物件類別不同，一個叫做 ifstream，一個叫 ofstream，但他們的操作方式是一樣的喔，我就不再撰寫了。


## 一些小補充

可以先看一下 [ifstream](https://www.cplusplus.com/reference/fstream/ifstream/) 跟 [ofstream](https://www.cplusplus.com/reference/fstream/ofstream/) 兩個類別有哪些方法可以用，你應該會看到前面有提到的 good、fail 之類的。

### [](#字元與字串讀入與寫出 "字元與字串讀入與寫出")字元與字串讀入與寫出

在做檔案讀入的時候，有時候我們可能只要讀，所以其實不用有特別的數字處理，我們有一些很常見的讀入方式。

#### [](#針對讀入 "針對讀入")針對讀入

- 字元讀入(一次讀一個字)，當遇到終止符停止，預設終止符為換行符號
    - [get 函式看這邊](https://www.cplusplus.com/reference/istream/istream/get/)

    ``` cpp
    int main(){
    ifstream in;
    in.open("data.txt");
    char c = ' ';
    while(in.get(c))
        cout << c;
    return 0;
    }
    ```
- 一次讀入一行

- [getline 看這邊](https://www.cplusplus.com/reference/istream/istream/getline/)
- getline 會視換行符號為一個斷點，所以換行輸出不出來
- 你必須手動換行
	-   這邊第一個參數是指標，所以你必須傳入一個記憶體位置，所以這邊第一個參數通常會用陣列(指標之後會教)
	- 第二個參數是你要接收的數量，這邊寫 256 但其實只會接收 255 個，因為在這邊最後一個會是結束字元 `\0`，沒有為甚麼。
	- 而 overloading function 有三個參數的，第三個參數是終止字符，代表說我們讀入時，遇到哪個字就結束，比方說我們設定大寫 `A`，那我們讀入到大寫 `A` 就會結束這個 getline 函式。
	- 而只有兩個參數的 getline 他預設是遇到換行符號 `\n` 就會結束這個函式。
	
    ``` cpp
	int main(){
    ifstream in;
    in.open("data.txt");
    char c[256];
    while(in.getline(c, 256)){ //第二個參數必須要先宣告一個陣列才行
        cout << c << endl;        
    }
    return 0;
    }
    ```

#### 針對寫出

針對寫出，只有 put 一個字一個字輸出的方式而已，沒有像 getline 這樣的東西，當然有其他輸出的方式，但就讓大家自己去花心思了解摟，舉例 write 函式之類的。

- 一次寫出一個字
    - [put 看這邊](https://www.cplusplus.com/reference/ostream/ostream/put/)
    - 你會發現這邊是寫 `cin.get(c)`，cin 本身就是做使用者輸入，而 get 是每次讀取一個字，然後讀取使用者輸入的這個字存給 c 變數。
    - `out.put(c)`就把這個 c 變數的值放進去檔案內
    
    ``` cpp
    int main(){
    ofstream out;
    out.open("input.txt");
    char c = ' ';
    do{
        cin.get(c); // 你也可以寫 c = cin.get();
                    // 因為是一次讀入一個字，你的 c 不能夠宣告成陣列喔
        out.put(c);
    }while(c != '.');
    }
    ```

    使用 get 時會遇到的換行問題
    不囉嗦，先看例子。

    ``` cpp
    int main(){
    int number = 0;
    cout << "plz input a number!" << endl;
    cin >> number; //假設輸入 3 後按 enter
    char symbol;
    cout << "plz input a symbol!" << endl;
    cin.get(symbol); // 假設輸入 A
    }
    ```

    可能期望上面的 CODE 會印出如下內容:

    ``` bash
    plz input a number!
    3
    plz input a symbol!
    A
    ```

    上面的 CODE 會做出下面的舉動:

    ``` bash
    plz input a number!
    3
    plz input a symbol!
                (其實這邊是換行符號!A沒辦法輸入)
    ```

    這個時候要來談談緩衝區了(input stream, 又稱 Buffer)
    當我們在第四行輸入數字後按下 \n 其實這個換行符號也會視為一個字，但因為他的型態不是數字，所以 number 沒有把它吃走，而這個換行符號就留在了所謂的緩衝區裡面，等待下一個符合型態的跟 cin 相關的輸入來把他帶走，而第七行 cin.get 會取走一個字元，因為緩衝區有一個換行符號在，所以這個 symbol 會把這個換行符號帶走，自然你們就無法輸入了。

- 第一種解法，在 cin>>number 後面多 cin.get(symbol) 一次
``` cpp
#include <iostream>
#include <string>
using namespace std;
int main {
int number = 0;
cout << "plz input a number"<< eddl;
cin >> number //假設輸入3後按enter
char symbol;
cout << "plz input a symbol" << endl;
cin.get(symbol);//假設輸入A
return 0;
}
```

    這樣上面的 cin.get(symbol) 就會先吃掉換行符號

- 第二種解法，呼叫 cin.ignore() 來清空緩衝區，這個函式非常好用，他能夠把緩衝區那些被遺留下來的字清理掉。

``` cpp
#include <iostream>
#include <string>
using namespace std;
int main {
int number = 0;
cout << "plz input a number"<< eddl;
cin >> number //假設輸入3後按enter
char symbol;
cout << "plz input a symbol" << endl;
cin.ignore();
cin.get(symbol);//假設輸入A
return 0;
}```

### ifstream 與 ofstream 作為參數
講到作為參數，相信現在大家都能馬上想到 function 了，至於要使用 function 代表我們一定想拆分什麼功能出來，不要讓程式碼都寫在 main 裡面，提高程式利用率。

``` cpp
void readwrite_INT_fromFile(ifstream &fin, ofstream &fout){
    int value = 0;
    while(!fin.eof()){
        fin >> value;
        if(!fin.fail()){
            cout << value << endl;
            fout << value << endl;
        }
    }
}
int main(){
    ifstream in;
    ofstream out;
    in.open("data.txt");
    out.open("input.txt");
    readwrite_INT_fromFile(in, out);
    in.close();
    out.close();
    return 0;
}
```


還記得在參數內我有說過，你是明確定義一個 function 的藍圖(規格)，告訴電腦我們需要傳入什麼，常見的有 int、string、char、double、float。

但今天我們想要把讀資料這件事情拆分給函式來完成，第 12 行先建立資料讀入物件，然後呼叫我們設計的讀資料函式並把這個讀入串流當作參數傳進函式內。
到第一行開始執行我們的函式內容。

這邊你會問，為什麼要 call-by reference?

幫大家快速複習，傳參考是為 actual parameter 建立一個參考變數，所以會有下面這樣的操作發生。
ifstream &fin = in;

還記得 call-by reference 當初說的時候能夠讓我們操控同樣一塊記憶體，減少記憶體的使用，避免不必要空間的浪費。

代表說 in、out 這兩個參數也會需要記憶體哦，然而讀入與寫出串流其實在整份程式裡面可以只各<span style="color:#ff0000;">依靠一個物件就能夠全部完成</span>，也就是第 12 行的 in 跟 第 13 行的 out，我們可以利用 in 開啟很多檔案，讀入很多資料，反之。
既然整個程式內只要一份，那為什麼還要多開記憶體呢? 所以這是為什麼使用 call-by reference 的原因。
<span style="color:#ff0000;">而 ifstream 或 ofstream 當參數時，一定要是 call-by reference</span>

小總結
在 C++ 內，<span style="color:#ff0000;">串流物件當作參數時務必要為參考變數</span>，也就是一定要使用 call-by reference 來做參數傳遞。<span style="color:#ff0000;">目的是為了不浪費記憶體空間。</span>

### 那什麼是只要一份??
甚麼叫做只要一份? 我們看一下下面的範例。

``` cpp
int main(){
    ifstream in;
    in.open("test.txt");
    int a, b , c;
    in >> a >> b >> c; //其實下面的註解也是一樣的意思
    /*
        in >> a;
        in >> b;
        in >> c;
    */
}
```

在上面的這個小範例，註解內的內容我們可以寫成第五行一次完成，然而在註解內的方式出現三次 in，他們其實都是相同的 in 物件哦，因為我們只有宣告一個 ifstream 物件叫做 in，他<span style="color:#FF0000">一個人負責檔案讀入的全部事務。</span>

那如果有兩個檔案，都各有五個數字，我想把第一個檔案的第一個數字跟第二個檔案的第一個數字相加，依此類推的話，那怎麼辦?
只有一個讀入物件的話你可以利用陣列。

``` cpp
int main(){
    ifstream in;
    in.open("test1.txt");
    int arr1[5], a;
    for(int i = 0; i < 5; i++){
        in >> a; //從檔案讀入一個數字
        arr1[i] = a;// assign 給陣列
    }
    in.close();
    int arr2[5];
    in.open("test2.txt");
    for(int i = 0; i < 5; i++){
        in >> a;
        arr2[i] = a;
    }
    for(int i = 0; i < 5; i++){
        cout << arr1[i] + arr2[i] << endl;
    }
    return 0;
}
```

那你可能會覺得這樣有點費功夫，能不能再多一個讀入物件，來負責讀入 test2.txt 呢? 這是絕對沒問題的。

``` cpp
int main(){
    ifstream in1, in2;
    in1.open("test1.txt");
    in2.open("test2.txt");
    int a, b;
    for(int i = 0; i < 5; i++){
        in1 >> a;
        in2 >> b;
        cout << a + b << endl;
    }
    in1.close();
    in2.close();
    return 0;
}
```

這樣寫相對就簡單很多，所以你說到底甚麼時候要多宣告讀入(或寫出)物件呢?
<span style="color:#FF0000">完全端看你的需求而定!!</span>

