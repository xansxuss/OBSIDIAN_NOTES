    <!DOCTYPE html>
<html lang="zh-CN">

<head>
    <meta charset="utf-8">
    <link rel="canonical" href="https://blog.csdn.net/hexiaolong2009/article/details/102596845"/>
    <meta http-equiv="content-type" content="text/html; charset=utf-8">
    <meta name="renderer" content="webkit"/>
    <meta name="force-rendering" content="webkit"/>
    <meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, minimum-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="report" content='{"pid": "blog", "spm":"1001.2101"}'>
    <meta name="referrer" content="always">
    <meta http-equiv="Cache-Control" content="no-siteapp" /><link rel="alternate" media="handheld" href="#" />
    <meta name="shenma-site-verification" content="5a59773ab8077d4a62bf469ab966a63b_1497598848">
    <meta name="applicable-device" content="pc">
    <link  href="https://g.csdnimg.cn/static/logo/favicon32.ico"  rel="shortcut icon" type="image/x-icon" />
    <title>dma-buf 由浅入深（七） —— alloc page 版本_dma buf begin cpu access  dma bidirectional-CSDN博客</title>
    <script>
      (function(){ 
        var el = document.createElement("script"); 
        el.src = "https://s3a.pstatp.com/toutiao/push.js?1abfa13dfe74d72d41d83c86d240de427e7cac50c51ead53b2e79d40c7952a23ed7716d05b4a0f683a653eab3e214672511de2457e74e99286eb2c33f4428830"; 
        el.id = "ttzz"; 
        var s = document.getElementsByTagName("script")[0]; 
        s.parentNode.insertBefore(el, s);
      })(window)
    </script>
        <meta name="keywords" content="dma buf begin cpu access  dma bidirectional">
        <meta name="csdn-baidu-search"  content='{"autorun":true,"install":true,"keyword":"dma buf begin cpu access  dma bidirectional"}'>
    <meta name="description" content="文章浏览阅读1.3w次，点赞13次，收藏15次。本文深入探讨了DMA-BUF驱动程序中使用alloc_page()替代kzalloc()进行内存分配的方法，详细比较了两种方式在DMA-BUF操作中的具体实现，并提供了exporter和importer驱动的示例代码。">
              <link rel="stylesheet" type="text/css" href="https://csdnimg.cn/release/blogv2/dist/pc/css/detail_enter-8834632c0c.min.css">
    <style>
        #content_views pre{
            -webkit-touch-callout: none;
            -webkit-user-select: none;
            -khtml-user-select: none;
            -moz-user-select: none;
            -ms-user-select: none; 
            user-select: none; 
        }
        #content_views pre code{
            -webkit-touch-callout: none;
            -webkit-user-select: none;
            -khtml-user-select: none;
            -moz-user-select: none;
            -ms-user-select: none; 
            user-select: none; 
        }
    </style>
    <script type="application/ld+json">{"@context":"https://ziyuan.baidu.com/contexts/cambrian.jsonld","@id":"https://blog.csdn.net/hexiaolong2009/article/details/102596845","appid":"1638831770136827","pubDate":"2020-01-12T20:06:09","title":"dma-buf 由浅入深（七） &mdash;&mdash; alloc page 版本_dma buf begin cpu access  dma bidirectional-CSDN博客","upDate":"2020-01-12T20:06:09"}</script>
        <link rel="stylesheet" type="text/css" href="https://csdnimg.cn/release/blogv2/dist/pc/themesSkin/skin-yellow/skin-yellow-28d34ab5fa.min.css">
    <script src="https://g.csdnimg.cn/lib/jquery/1.12.4/jquery.min.js" type="text/javascript"></script>
    <script src="https://g.csdnimg.cn/lib/jquery-migrate/1.4.1/jquery-migrate.js" type="text/javascript"></script>
    <script type="text/javascript">
        var isCorporate = false;
        var username =  "hexiaolong2009";
        var skinImg = "white";

        var blog_address = "https://blog.csdn.net/hexiaolong2009";
        var currentUserName = "";
        var isOwner = false;
        var loginUrl = "http://passport.csdn.net/account/login?from=https://blog.csdn.net/hexiaolong2009/article/details/102596845";
        var blogUrl = "https://blog.csdn.net/";
        var starMapUrl = "https://ai.csdn.net";
        var inscodeHost = "https://inscode.csdn.net";
        var paymentBalanceUrl = "https://csdnimg.cn/release/vip-business-components/vipPaymentBalance.js";
        var appBlogDomain = "https://app-blog.csdn.net";
        var avatar = "https://profile-avatar.csdnimg.cn/13e0f6961a114b5cb09ff65ed930a667_hexiaolong2009.jpg!1";
        var isCJBlog = false;
        var isStarMap = false;
        var articleTitle = "dma-buf 由浅入深（七） —— alloc page 版本";
        var articleDesc = "文章浏览阅读1.3w次，点赞13次，收藏15次。本文深入探讨了DMA-BUF驱动程序中使用alloc_page()替代kzalloc()进行内存分配的方法，详细比较了两种方式在DMA-BUF操作中的具体实现，并提供了exporter和importer驱动的示例代码。";
        var articleTitles = "dma-buf 由浅入深（七） —— alloc page 版本_dma buf begin cpu access  dma bidirectional-CSDN博客";
        var nickName = "何小龙";
        var articleDetailUrl = "https://blog.csdn.net/hexiaolong2009/article/details/102596845";
        var vipUrlV = "https://mall.csdn.net/vip?vipSource=learningVip";
        if(window.location.host.split('.').length == 3) {
            blog_address = blogUrl + username;
        }
        var skinStatus = "White";
        var blogStaticHost = "https://csdnimg.cn/release/blogv2/"
          var payColumn = false
    </script>
        <meta name="toolbar" content='{"type":"0","fixModel":"1"}'>
    <script src="https://g.csdnimg.cn/??fixed-sidebar/1.1.7/fixed-sidebar.js" type="text/javascript"></script>
    <script src='//g.csdnimg.cn/common/csdn-report/report.js' type='text/javascript'></script>
    <link rel="stylesheet" type="text/css" href="https://csdnimg.cn/public/sandalstrap/1.4/css/sandalstrap.min.css">
    <style>
        .MathJax, .MathJax_Message, .MathJax_Preview{
            display: none
        }
    </style>
    <script src="https://dup.baidustatic.com/js/ds.js"></script>
      <script type="text/javascript">
        (function(c,l,a,r,i,t,y){
            c[a]=c[a]||function(){(c[a].q=c[a].q||[]).push(arguments)};
            t=l.createElement(r);t.async=1;t.src="https://www.clarity.ms/tag/"+i;
            y=l.getElementsByTagName(r)[0];y.parentNode.insertBefore(t,y);
        })(window, document, "clarity", "script", "lgtpix6r85");
      </script>
    <script src="/cdn_cgi_bs_bot/static/crypto.min.js"></script>
    <script src="/cdn_cgi_bs_bot/static/bot-score-v1.js"></script>
    <script src="/cdn_cgi_bs_captcha/static/js/waf_captcha_embedded_bs.js"></script>
</head>
  <body class="nodata  " style="">
    <div id="toolbarBox" style="min-height: 48px;"></div>
        <script>
            var toolbarSearchExt = '{\"id\":102596845,\"landingWord\":[\"dma buf begin cpu access  dma bidirectional\"],\"queryWord\":\"\",\"tag\":[\"dma-buf\",\"内核\",\"linux\"],\"title\":\"dma-buf 由浅入深（七） &mdash;&mdash; alloc page 版本\"}';
        </script>
      <script src="https://g.csdnimg.cn/common/csdn-toolbar/csdn-toolbar.js" type="text/javascript"></script>
    <script>
    (function(){
        var bp = document.createElement('script');
        var curProtocol = window.location.protocol.split(':')[0];
        if (curProtocol === 'https') {
            bp.src = 'https://zz.bdstatic.com/linksubmit/push.js';
        }
        else {
            bp.src = 'http://push.zhanzhang.baidu.com/push.js';
        }
        var s = document.getElementsByTagName("script")[0];
        s.parentNode.insertBefore(bp, s);
    })();
    </script>

    <link rel="stylesheet" href="https://csdnimg.cn/release/blogv2/dist/pc/css/blog_code-01256533b5.min.css">
    <link rel="stylesheet" href="https://csdnimg.cn/release/blogv2/dist/mdeditor/css/editerView/chart-3456820cac.css" />
    <link rel="stylesheet" href="https://g.csdnimg.cn/lib/swiper/6.0.4/css/swiper.css" />
    <script src="https://g.csdnimg.cn/lib/swiper/6.0.4/js/swiper.js" async></script>
    <script>
      var articleId = 102596845;
        var privateEduData = [];
        var privateData = ["示例代码","内存","dma"];//高亮数组
      var crytojs = "https://csdnimg.cn/release/blogv2/dist/components/js/crytojs-ca5b8bf6ae.min.js";
      var commentscount = 6;
      var commentAuth = 2;
      var curentUrl = "https://blog.csdn.net/hexiaolong2009/article/details/102596845";
      var myUrl = "https://my.csdn.net/";
      var isGitCodeBlog = false;
      var vipActivityIcon = "https://i-operation.csdnimg.cn/images/df6c67fa661c48eba86beaeb64350df0.gif";
      var isOpenSourceBlog = false;
      var isVipArticle = false;
        var highlight = ["bidirectional","access","linux","alloc","begin","由浅入深","page","cpu","buf","dma","版本","内核","七","(",")","-"];//高亮数组
        var isRecommendModule = true;
          var isBaiduPre = true;
          var baiduCount = 2;
          var setBaiduJsCount = 10;
        var viewCountFormat = 13105;
      var share_card_url = "https://app-blog.csdn.net/share?article_id=102596845&username=hexiaolong2009"
      var mallVipUrl = "https://mall.csdn.net/vip?vipSource=article"
      var vipArticleAbStyle = "t_1"
      var vipArticleCpStyle = "t_1"
      var detailheaderAbCommunity = "control"
      var codeAiAbStyle = "exp1"
      var runProjectDeepBlogAB = "isRunProject is false"
      var gitcodeHighlightWordAB = "exp1"
      var deepblogUrl = ""
        deepblogUrl = "https://deepblog.net?utm_source=blog_onekey_run";
      var codeAiAbObjStyle = ""
        codeAiAbObjStyle = "{\"control\":{\"title\":\"AI写代码\",\"destUrl\":\"https://trae.com.cn?utm_source=community&utm_medium=csdn&utm_campaign=daima\",\"imgUrl\":\"https://i-operation.csdnimg.cn/images/a5fff6f6c9f0464c9a46b130c972952b.png\"},\"exp1\":{\"title\":\"一键获取完整项目代码\",\"runClose\":true,\"destUrl\":\"https://inscode.net?utm_source=blog_code_block\",\"imgUrl\":\"https://i-operation.csdnimg.cn/images/25e1eba3e6bc4df7ba20f2b6011fbe21.png\",\"blogUrl\":true},\"control_run_project\": {\"title\": \"运行项目并下载源码\",\"destUrl\": \"\",\"imgUrl\":\"https://i-operation.csdnimg.cn/images/46c457a2cf8b4b9b8f17a2ab71461d4a.png\"},\"control_deepblog\": {\"title\": \"AI生成项目\",\"destUrl\": \"https://inscode.net?utm_source=blog_code_block_fixed\",\"imgUrl\":\"https://i-operation.csdnimg.cn/images/9899ea0f099e4e4e8b1fcdb918a27fcd.png\",\"blogUrl\": true},\"exp2\":{\"title\":\"智能体编程\",\"destUrl\":\"https://qoder.com/referral?referral_code=kyKxftaZjisNKOow777DARC0j35axVBq\",\"imgUrl\":\"https://i-operation.csdnimg.cn/images/afa15dc565924b96a783f4db37687511.png\"},\"exp3\":{\"title\":\"AI构建项目\",\"destUrl\":\"https://t.csdnimg.cn/9I17\",\"imgUrl\":\"https://i-operation.csdnimg.cn/images/bf18ebff2c3748d59ea0f95954bb4b4a.png\"}}";
      var aiSideSegment = -1;
        aiSideSegment = "16";
      var articleType = 1;
      var baiduKey = "dma buf begin cpu access  dma bidirectional";
      var copyPopSwitch = true;
      var needInsertBaidu = true;
      var recommendRegularDomainArr = ["blog.csdn.net/.+/article/details/","download.csdn.net/download/","edu.csdn.net/course/detail/","ask.csdn.net/questions/","bbs.csdn.net/topics/","www.csdn.net/gather_.+/"]
      var codeStyle = "atom-one-dark";
      var baiduSearchType = "baidulandingword";
      var sharData = "{\"hot\":[{\"id\":1,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/a5f4260710904e538002a6ab337939b3.png\"},{\"id\":2,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/188b37199a2c4b74b1d9ffc39e0d52de.png\"},{\"id\":3,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/14ded358b631444581edd98a256bc5af.png\"},{\"id\":4,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/1470f23a770444d986ad551b9c33c5be.png\"},{\"id\":5,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/c329f5181dc74f6c9bd28c982bb9f91d.png\"},{\"id\":6,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/ccd8a3305e81460f9c505c95b432a65f.png\"},{\"id\":7,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/bc89d8283389440d97fc4d30e30f45e1.png\"},{\"id\":8,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/452d485b4a654f5592390550d2445edf.png\"},{\"id\":9,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/f8b9939db2ed474a8f43a643015fc8b7.png\"},{\"id\":10,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/6de8864187ab4ed3b1db0856369c36ff.png\"},{\"id\":11,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/673cc3470ff74072acba958dc0c46e2d.png\"},{\"id\":12,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/930c119760ac4491804db80f9c6d4e3f.png\"},{\"id\":13,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/15e6befb05a24233bc2b65e96aa8d972.png\"},{\"id\":14,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/2075fd6822184b95a41e214de4daec13.png\"},{\"id\":15,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/859b1552db244eb6891a809263a5c657.png\"},{\"id\":16,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/0be2f920f1f74290a98921974a9613fd.png\"},{\"id\":17,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/2e97e00b43f14afab494ea55ef3f4a6e.png\"},{\"id\":18,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/ff4ab252f46e444686f5135d6ebbfec0.png\"},{\"id\":19,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/ae029bbe99564e79911657912d36524f.png\"},{\"id\":20,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/b3ece39963de440388728e9e7b9bf427.png\"},{\"id\":21,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/6f14651a99ba486e926d63b6fa692997.png\"},{\"id\":22,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/83ceddf050084875a341e32dcceca721.png\"},{\"id\":23,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/b90368b8fd5d4c6c8c79a707d877cf7c.png\"},{\"id\":24,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/aeffae14ecf14e079b2616528c9a393b.png\"},{\"id\":25,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/c5a06b5a13d44d16bed868fc3384897a.png\"},{\"id\":26,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/08b697658b844b318cea3b119e9541ef.png\"},{\"id\":27,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/68ccb0b8d09346ac961d2b5c1a8c77bf.png\"},{\"id\":28,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/a2227a247e37418cbe0ea972ba6a859b.png\"},{\"id\":29,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/3a42825fede748f9993e5bb844ad350d.png\"},{\"id\":30,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/8882abc1dd484224b636966ea38555c3.png\"},{\"id\":31,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/4f6a5f636a3e444d83cf8cc06d87a159.png\"},{\"id\":32,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/1953ef79c56b4407b78d7181bdff11c3.png\"},{\"id\":33,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/c04a2a4f772948ed85b5b0380ed36287.png\"},{\"id\":34,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/5b4fecd05091405ea04d8c0f53e9f2c7.png\"},{\"id\":35,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/b89f576d700344e280d6ceb2a66c2420.png\"},{\"id\":36,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/1c65780e11804bbd9971ebadb3d78bcf.png\"},{\"id\":37,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/d590db2055f345db9706eb68a7ec151a.png\"},{\"id\":38,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/fe602f80700b4f6fb3c4a9e4c135510e.png\"},{\"id\":39,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/39ff2fcd31e04feba301a071976a0ba7.png\"},{\"id\":40,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/f9b61b3d113f436b828631837f89fb39.png\"},{\"id\":41,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/df1aca5f610c4ad48cd16da88c9c8499.png\"},{\"id\":42,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/d7acf73a1e6b41399a77a85040e10961.png\"},{\"id\":43,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/b7f1b63542524b97962ff649ab4e7e23.png\"}],\"vip\":[{\"id\":1,\"vipUrl\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220920101150.png\",\"url\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220920101154.png\"},{\"id\":2,\"vipUrl\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220920101204.png\",\"url\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220920101208.png\"},{\"id\":3,\"vipUrl\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220920101211.png\",\"url\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220920101215.png\"},{\"id\":4,\"vipUrl\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220920101218.png\",\"url\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220920101220.png\"},{\"id\":5,\"vipUrl\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220920101223.png\",\"url\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220920101226.png\"},{\"id\":6,\"vipUrl\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220922100635.png\",\"url\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220922100639.png\"},{\"id\":7,\"vipUrl\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220922100642.png\",\"url\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220922100644.png\"},{\"id\":8,\"vipUrl\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220922100647.png\",\"url\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220922100649.png\"},{\"id\":9,\"vipUrl\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220922100652.png\",\"url\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220922100655.png\"},{\"id\":10,\"vipUrl\":\"https:\\/\\/img-blog.csdnimg.cn\\/55de67481fde4b04b97ad78f11fe369a.png\",\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/bb2418fb537e4d78b10d8765ccd810c5.png\"},{\"id\":11,\"vipUrl\":\"https:\\/\\/img-blog.csdnimg.cn\\/579c713394584d128104ef1044023954.png\",\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/f420d9fbcf5548079d31b5e809b6d6cd.png\"},{\"id\":12,\"vipUrl\":\"https:\\/\\/img-blog.csdnimg.cn\\/75b7f3155ba642f5a4cc16b7baf44122.png\",\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/a9030f5877be401f8b340b80b0d91e64.png\"},{\"id\":13,\"vipUrl\":\"https:\\/\\/img-blog.csdnimg.cn\\/0903d33cafa54934be3780aa54ae958d.png\",\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/2cd8c8929f5a42fca5da2a0aeb456203.png\"},{\"id\":14,\"vipUrl\":\"https:\\/\\/img-blog.csdnimg.cn\\/949fd7c22884439fbfc3c0e9c3b8dee7.png\",\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/dafbea9bd9eb4f3b962b48dc41657f89.png\"},{\"id\":15,\"vipUrl\":\"https:\\/\\/img-blog.csdnimg.cn\\/4119cfddd71d4e6a8a27a18dbb74d90e.png\",\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/c56310c8b6384d9e85388e4e342ce508.png\"},{\"id\":16,\"vipUrl\":\"https:\\/\\/img-blog.csdnimg.cn\\/121575274da142bcbbbbc2e8243dd411.png\",\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/5013993de06542f881018bb9abe2edf7.png\"},{\"id\":17,\"vipUrl\":\"https:\\/\\/img-blog.csdnimg.cn\\/4d97aa6dd4fe4f09a6bef5bdf8a6abcd.png\",\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/76f23877b6ad4066ad45ce8e31b4b977.png\"},{\"id\":18,\"vipUrl\":\"https:\\/\\/img-blog.csdnimg.cn\\/fdb619daf21b4c829de63b9ebc78859d.png\",\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/a1abe5d27a5441f599adfe662f510243.png\"},{\"id\":19,\"vipUrl\":\"https:\\/\\/img-blog.csdnimg.cn\\/676b7707bb11410f8f56bc0ed2b2345c.png\",\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/7ac5b467fbf24e1d8c2de3f3332c4f54.png\"},{\"id\":20,\"vipUrl\":\"https:\\/\\/img-blog.csdnimg.cn\\/0becb8cc227e4723b765bdd69a20fd4a.png\",\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/fdec85b26091486b9a89d0b8d45c3749.png\"},{\"id\":21,\"vipUrl\":\"https:\\/\\/img-blog.csdnimg.cn\\/1a6c06235ad44941b38c54cbc25a370c.png\",\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/410a06cda2d44b0c84578f88275caf70.png\"}],\"map\":{\"hot\":\"热门\",\"vip\":\"VIP\"}}";
        var commentListModule = "true"
      var canRead = true;
      var blogMoveHomeArticle = false;
      var showSearchText = "";
      var articleSource = 1;
      var articleReport = '{"pid": "blog", "spm":"1001.2101"}';
        var baiduSearchChannel = 'pc_relevant'
        var baiduSearchIdentification = '.235^v43^pc_blog_bottom_relevance_base9'
        var distRequestId = '1766382499823_87507'
        var initRewardObject = {
          giver: currentUserName,
          anchor: username,
          articleId: articleId,
          sign: ''
        }
        var isLikeStatus = false;
        var isUnLikeStatus = false;
        var studyLearnWord = "";
        var unUseCount = 0;
        var codeMaxSize = 0;
        var overCost = true;
        var isCurrentUserVip = false;
        var contentViewsHeight = 0;
        var contentViewsCount = 0;
        var contentViewsCountLimit = 5;
        var isShowConcision = false;
        var lastTime = 0
        var postTime = "2020-01-12 20:06:09"
      var isCookieConcision = false
      var isHasDirectoryModel = false
      var isShowSideModel = false
      var isShowDirectoryModel = true
      function getCookieConcision(sName){
        var allCookie = document.cookie.split("; ");
        for (var i=0; i < allCookie.length; i++){
          var aCrumb = allCookie[i].split("=");
          if (sName == aCrumb[0])
            return aCrumb[1];
        }
        return null;
      }
      if (getCookieConcision('blog_details_concision') && getCookieConcision('blog_details_concision') == 0){
        isCookieConcision = true
        isShowSideModel = true
        isShowDirectoryModel = false
      }
    </script>
        <div class="main_father clearfix d-flex justify-content-center " style="height:100%;">
          <div class="container clearfix " id="mainBox">
          <script>
          if (!isCookieConcision) {
            $('.main_father').removeClass('mainfather-concision')
            $('.main_father .container').removeClass('container-concision')
          } else {
            $('#mainBox').css('margin-right', '0')
          }
          </script>
          <main>
<script type="text/javascript">
    var resourceId =  "";
    function getQueryString(name) {   
      var reg = new RegExp("(^|&)" + name + "=([^&]*)(&|$)"); //构造一个含有目标参数的正则表达式对象  
      var r = window.location.search.substr(1).match(reg);  //匹配目标参数
      if( r != null ) return decodeURIComponent( r[2] ); return '';   
    }
    function stripscript(s){ 
      var pattern = new RegExp("[`~!@#$^&*()=|{}':;',\\[\\].<>/?~！@#￥……&*（）——|{}【】‘；：”“'。，、？%]") 
      var rs = ""; 
      for (var i = 0; i < s.length; i++) { 
        rs = rs+s.substr(i, 1).replace(pattern, ''); 
      } 
      return rs;
    }
    var blogHotWords = stripscript(getQueryString('utm_term')).length > 1 ? stripscript(getQueryString('utm_term')) : ''
</script>
<div class="blog-content-box">
  <div class="article-header-box" id="article-header-box">
    <div class="article-header">
      <div class="article-title-box">
        <h1 class="title-article" id="articleContentId">dma-buf 由浅入深（七） —— alloc page 版本</h1>
      </div>
      <div class="article-info-box">
              <div class="up-time">最新推荐文章于&nbsp;2025-10-14 15:13:29&nbsp;发布</div>
          <div class="article-bar-top">
              <div class="bar-content active">
              <span class="article-type-text original">原创</span>
                    <span class="time blog-postTime" data-time="2020-01-12 20:06:09">最新推荐文章于&nbsp;2025-10-14 15:13:29&nbsp;发布</span>
                <span class="border-dian">·</span>
                <span class="read-count">1.3w 阅读</span>
                <div class="read-count-box is-like like-ab-new" data-type="top">
                  <span class="border-dian">·</span>
                  <img class="article-read-img article-heard-img active" style="display:none" id="is-like-imgactive-new" src="https://csdnimg.cn/release/blogv2/dist/pc/img/newHeart2023Active.png" alt="">
                  <img class="article-read-img article-heard-img" style="display:block" id="is-like-img-new" src="https://csdnimg.cn/release/blogv2/dist/pc/img/newHeart2023Black.png" alt="">
                  <span class="read-count" id="blog-digg-num" style="color:;">
                      13
                  </span>
                </div>
                <span class="border-dian">·</span>
                <a id="blog_detail_zk_collection" class="un-collection" data-report-click='{"mod":"popu_823","spm":"1001.2101.3001.4232","ab":"new"}'>
                  <img class="article-collect-img article-heard-img un-collect-status isdefault" style="display:inline-block" src="https://csdnimg.cn/release/blogv2/dist/pc/img/tobarCollect2.png" alt="">
                  <img class="article-collect-img article-heard-img collect-status isactive" style="display:none" src="https://csdnimg.cn/release/blogv2/dist/pc/img/tobarCollectionActive2.png" alt="">
                  <span class="get-collection">
                      15
                  </span>
                </a>

                  <span class="border-dian">·</span>
                  <div class="href-article-edit-new">
                    <span class="href-article-edit-click">CC 4.0 BY-SA版权</span>
                    <div class="slide-content-box-new">
                                版权声明：本文为博主原创文章，遵循<a href="http://creativecommons.org/licenses/by-sa/4.0/" target="_blank" rel="noopener"> CC 4.0 BY-SA </a>版权协议，转载请附上原文出处链接和本声明。
                    </div>
                  </div>
              </div>
              <div class="operating active">
              </div>
          </div>
          <div class="blog-tags-box">
             
              <div class="tags-box artic-tag-box">
                     <div class="article-tag">
                       <span class="label">文章标签：</span>
                      <p>
                          <a rel="nofollow" data-report-query="spm=1001.2101.3001.4223" data-report-click='{"mod":"popu_626","spm":"1001.2101.3001.4223","strategy":"dma-buf","ab":"new","extra":"{\"searchword\":\"dma-buf\"}"}' data-report-view='{"mod":"popu_626","spm":"1001.2101.3001.4223","strategy":"dma-buf","ab":"new","extra":"{\"searchword\":\"dma-buf\"}"}' class="tag-link-new" href="https://so.csdn.net/so/search/s.do?q=dma-buf&amp;t=all&amp;o=vip&amp;s=&amp;l=&amp;f=&amp;viparticle=&amp;from_tracking_code=tag_word&amp;from_code=app_blog_art" target="_blank" rel="noopener">#dma-buf</a>
                          <a rel="nofollow" data-report-query="spm=1001.2101.3001.4223" data-report-click='{"mod":"popu_626","spm":"1001.2101.3001.4223","strategy":"内核","ab":"new","extra":"{\"searchword\":\"内核\"}"}' data-report-view='{"mod":"popu_626","spm":"1001.2101.3001.4223","strategy":"内核","ab":"new","extra":"{\"searchword\":\"内核\"}"}' class="tag-link-new" href="https://so.csdn.net/so/search/s.do?q=%E5%86%85%E6%A0%B8&amp;t=all&amp;o=vip&amp;s=&amp;l=&amp;f=&amp;viparticle=&amp;from_tracking_code=tag_word&amp;from_code=app_blog_art" target="_blank" rel="noopener">#内核</a>
                          <a rel="nofollow" data-report-query="spm=1001.2101.3001.4223" data-report-click='{"mod":"popu_626","spm":"1001.2101.3001.4223","strategy":"linux","ab":"new","extra":"{\"searchword\":\"linux\"}"}' data-report-view='{"mod":"popu_626","spm":"1001.2101.3001.4223","strategy":"linux","ab":"new","extra":"{\"searchword\":\"linux\"}"}' class="tag-link-new" href="https://so.csdn.net/so/search/s.do?q=linux&amp;t=all&amp;o=vip&amp;s=&amp;l=&amp;f=&amp;viparticle=&amp;from_tracking_code=tag_word&amp;from_code=app_blog_art" target="_blank" rel="noopener">#linux</a>
                      </p>
                     </div>
                  <p class="community-name" id="community-name"></p>
              </div>
          </div>
       
          
       
      </div>
    </div>
  </div>
    <div id="blogHuaweiyunAdvert" class=""></div>
        <div id="blogColumnPayAdvert" class="">
              <div class="column-group">
                <div class="column-group-item column-group0 ">
                    <div class="item-l">
                        <a class="item-target" href="https://blog.csdn.net/hexiaolong2009/category_9281458.html" target="_blank" title="DRM (Direct Rendering Manager)"
                        data-report-view='{"spm":"1001.2101.3001.6332"}'
                        data-report-click='{"spm":"1001.2101.3001.6332"}'>
                            <img class="item-target" src="https://i-blog.csdnimg.cn/blog_column_migrate/c03d60f8b24740d58f94a787024155be.png?x-oss-process=image/resize,m_fixed,h_224,w_224" alt="">
                            <span class="title item-target">
                                <span>
                                <span class="tit">DRM (Direct Rendering Manager)</span>
                                    <span class="dec more">同时被 2 个专栏收录<img src="https://csdnimg.cn/release/blogv2/dist/pc/img/newArrowDown1White.png" alt=""></span>
                                </span>
                            </span>
                        </a>
                    </div>
                    <div class="item-m">
                        <span>29 篇文章</span>
                    </div>
                    <div class="item-r">
                            <a class="item-target article-column-bt articleColumnFreeBt" data-id="9281458">订阅专栏</a>
                    </div>
                </div>
                <div class="column-group-item column-group1 ">
                    <div class="item-l">
                        <a class="item-target" href="https://blog.csdn.net/hexiaolong2009/category_10838100.html" target="_blank" title="DMA-BUF"
                        data-report-view='{"spm":"1001.2101.3001.6332"}'
                        data-report-click='{"spm":"1001.2101.3001.6332"}'>
                            <img class="item-target" src="https://i-blog.csdnimg.cn/columns/default/20201014180756923.png?x-oss-process=image/resize,m_fixed,h_224,w_224" alt="">
                            <span class="title item-target">
                                <span>
                                <span class="tit">DMA-BUF</span>
                                </span>
                            </span>
                        </a>
                    </div>
                    <div class="item-m">
                        <span>10 篇文章</span>
                    </div>
                    <div class="item-r">
                            <a class="item-target article-column-bt articleColumnFreeBt" data-id="10838100">订阅专栏</a>
                    </div>
                </div>
              </div>

        </div>
      <div class="ai-abstract-box">
         <div class="ai-abstract">
          <div class="abstract-content">
            <img class="lock-img" src="https://i-operation.csdnimg.cn/images/a7311a21245d4888a669ca3155f1f4e5.png" alt="">本文深入探讨了DMA-BUF驱动程序中使用alloc_page()替代kzalloc()进行内存分配的方法，详细比较了两种方式在DMA-BUF操作中的具体实现，并提供了exporter和importer驱动的示例代码。
          </div>
        </div>
      </div>
    <article class="baidu_pl">
        <div id="article_content" class="article_content clearfix">
        <link rel="stylesheet" href="https://csdnimg.cn/release/blogv2/dist/mdeditor/css/editerView/kdoc_html_views-1a98987dfd.css">
        <link rel="stylesheet" href="https://csdnimg.cn/release/blogv2/dist/mdeditor/css/editerView/ck_htmledit_views-10bf609291.css">
                <div id="content_views" class="markdown_views prism-atom-one-dark">
                    <svg xmlns="http://www.w3.org/2000/svg" style="display: none;">
                        <path stroke-linecap="round" d="M5,0 0,2.5 5,5z" id="raphael-marker-block" style="-webkit-tap-highlight-color: rgba(0, 0, 0, 0);"></path>
                    </svg>
                    <p><a href="https://blog.csdn.net/hexiaolong2009/article/details/102596744">dma-buf 由浅入深&#xff08;一&#xff09; —— 最简单的 dma-buf 驱动程序</a><br /> <a href="https://blog.csdn.net/hexiaolong2009/article/details/102596761">dma-buf 由浅入深&#xff08;二&#xff09; —— kmap / vmap</a><br /> <a href="https://blog.csdn.net/hexiaolong2009/article/details/102596772">dma-buf 由浅入深&#xff08;三&#xff09; —— map attachment</a><br /> <a href="https://blog.csdn.net/hexiaolong2009/article/details/102596791">dma-buf 由浅入深&#xff08;四&#xff09; —— mmap</a><br /> <a href="https://blog.csdn.net/hexiaolong2009/article/details/102596802">dma-buf 由浅入深&#xff08;五&#xff09; —— File</a><br /> <a href="https://blog.csdn.net/hexiaolong2009/article/details/102596825">dma-buf 由浅入深&#xff08;六&#xff09; —— begin / end cpu_access</a><br /> <a href="https://blog.csdn.net/hexiaolong2009/article/details/102596845">dma-buf 由浅入深&#xff08;七&#xff09; —— alloc page 版本</a><br /> <a href="https://blog.csdn.net/hexiaolong2009/article/details/103795381">dma-buf 由浅入深&#xff08;八&#xff09; —— ION 简化版</a></p> 
<hr /> 
<h3><a id="_11"></a>前言</h3> 
<p>在前面的 dma-buf 系列文章中&#xff0c;exporter 所分配的内存都是通过 kzalloc() 来分配的。本篇我们换个方式&#xff0c;使用 alloc_page() 来分配内存。</p> 
<h3><a id="_14"></a>对比</h3> 
<p>与之前的 kzalloc 方式相比&#xff0c;alloc_page 方式的主要区别如下&#xff1a;</p> 
<table><thead><tr><th align="left">dma_buf_ops</th><th align="left">kzalloc 方式</th><th align="left">alloc_page 方式</th></tr></thead><tbody><tr><td align="left">map_dma_buf</td><td align="left">dma_map_single()</td><td align="left">dma_map_page()</td></tr><tr><td align="left">unmap_dma_buf</td><td align="left">dma_unmap_single()</td><td align="left">dma_unmap_page()</td></tr><tr><td align="left">begin_cpu_access</td><td align="left">dma_sync_single_for_cpu()</td><td align="left">dma_sync_sg_for_cpu()</td></tr><tr><td align="left">end_cpu_access</td><td align="left">dma_sync_single_for_device()</td><td align="left">dma_sync_sg_for_device()</td></tr><tr><td align="left">kmap</td><td align="left">return dmabuf-&gt;priv;</td><td align="left">kmap()</td></tr><tr><td align="left">kmap_atomic</td><td align="left">return dmabuf-&gt;priv;</td><td align="left">kmap_atomic()</td></tr><tr><td align="left">vmap</td><td align="left">return dmabuf-&gt;priv;</td><td align="left">vmap()</td></tr><tr><td align="left">release</td><td align="left">kfree()</td><td align="left">put_page()</td></tr></tbody></table>
<h3><a id="_29"></a>示例</h3> 
<p><strong>exporter 驱动</strong><br /> 结合前面几篇文章的示例代码&#xff0c;将 dma_buf_ops 全部替换成 page 方式。</p> 
<p><a href="https://github.com/hexiaolong2008/sample-code/blob/master/dma-buf/08/exporter-page.c">exporter-page.c</a></p> 
<pre><code class="prism language-c"><span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/dma-buf.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/highmem.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/module.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/slab.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/miscdevice.h&gt;</span></span>

<span class="token keyword">static</span> <span class="token keyword">struct</span> dma_buf <span class="token operator">*</span>dmabuf_exported<span class="token punctuation">;</span>

<span class="token keyword">static</span> <span class="token keyword">int</span> <span class="token function">exporter_attach</span><span class="token punctuation">(</span><span class="token keyword">struct</span> dma_buf <span class="token operator">*</span>dmabuf<span class="token punctuation">,</span> <span class="token keyword">struct</span> device <span class="token operator">*</span>dev<span class="token punctuation">,</span>
			<span class="token keyword">struct</span> dma_buf_attachment <span class="token operator">*</span>attachment<span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token function">pr_info</span><span class="token punctuation">(</span><span class="token string">&#34;dmabuf attach device: %s\n&#34;</span><span class="token punctuation">,</span> <span class="token function">dev_name</span><span class="token punctuation">(</span>dev<span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token keyword">return</span> <span class="token number">0</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token keyword">static</span> <span class="token keyword">void</span> <span class="token function">exporter_detach</span><span class="token punctuation">(</span><span class="token keyword">struct</span> dma_buf <span class="token operator">*</span>dmabuf<span class="token punctuation">,</span> <span class="token keyword">struct</span> dma_buf_attachment <span class="token operator">*</span>attachment<span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token function">pr_info</span><span class="token punctuation">(</span><span class="token string">&#34;dmabuf detach device: %s\n&#34;</span><span class="token punctuation">,</span> <span class="token function">dev_name</span><span class="token punctuation">(</span>attachment<span class="token operator">-&gt;</span>dev<span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token keyword">static</span> <span class="token keyword">struct</span> sg_table <span class="token operator">*</span><span class="token function">exporter_map_dma_buf</span><span class="token punctuation">(</span><span class="token keyword">struct</span> dma_buf_attachment <span class="token operator">*</span>attachment<span class="token punctuation">,</span>
					 <span class="token keyword">enum</span> dma_data_direction dir<span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token keyword">struct</span> page <span class="token operator">*</span>page <span class="token operator">&#61;</span> attachment<span class="token operator">-&gt;</span>dmabuf<span class="token operator">-&gt;</span>priv<span class="token punctuation">;</span>
	<span class="token keyword">struct</span> sg_table <span class="token operator">*</span>table<span class="token punctuation">;</span>

	table <span class="token operator">&#61;</span> <span class="token function">kmalloc</span><span class="token punctuation">(</span><span class="token keyword">sizeof</span><span class="token punctuation">(</span><span class="token operator">*</span>table<span class="token punctuation">)</span><span class="token punctuation">,</span> GFP_KERNEL<span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token function">sg_alloc_table</span><span class="token punctuation">(</span>table<span class="token punctuation">,</span> <span class="token number">1</span><span class="token punctuation">,</span> GFP_KERNEL<span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token function">sg_set_page</span><span class="token punctuation">(</span>table<span class="token operator">-&gt;</span>sgl<span class="token punctuation">,</span> page<span class="token punctuation">,</span> PAGE_SIZE<span class="token punctuation">,</span> <span class="token number">0</span><span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token function">sg_dma_address</span><span class="token punctuation">(</span>table<span class="token operator">-&gt;</span>sgl<span class="token punctuation">)</span> <span class="token operator">&#61;</span> <span class="token function">dma_map_page</span><span class="token punctuation">(</span><span class="token constant">NULL</span><span class="token punctuation">,</span> page<span class="token punctuation">,</span> <span class="token number">0</span><span class="token punctuation">,</span> PAGE_SIZE<span class="token punctuation">,</span> dir<span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token keyword">return</span> table<span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token keyword">static</span> <span class="token keyword">void</span> <span class="token function">exporter_unmap_dma_buf</span><span class="token punctuation">(</span><span class="token keyword">struct</span> dma_buf_attachment <span class="token operator">*</span>attachment<span class="token punctuation">,</span>
			       <span class="token keyword">struct</span> sg_table <span class="token operator">*</span>table<span class="token punctuation">,</span>
			       <span class="token keyword">enum</span> dma_data_direction dir<span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token function">dma_unmap_page</span><span class="token punctuation">(</span><span class="token constant">NULL</span><span class="token punctuation">,</span> <span class="token function">sg_dma_address</span><span class="token punctuation">(</span>table<span class="token operator">-&gt;</span>sgl<span class="token punctuation">)</span><span class="token punctuation">,</span> PAGE_SIZE<span class="token punctuation">,</span> dir<span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token function">sg_free_table</span><span class="token punctuation">(</span>table<span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token function">kfree</span><span class="token punctuation">(</span>table<span class="token punctuation">)</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token keyword">static</span> <span class="token keyword">void</span> <span class="token function">exporter_release</span><span class="token punctuation">(</span><span class="token keyword">struct</span> dma_buf <span class="token operator">*</span>dma_buf<span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token keyword">struct</span> page <span class="token operator">*</span>page <span class="token operator">&#61;</span> dma_buf<span class="token operator">-&gt;</span>priv<span class="token punctuation">;</span>

	<span class="token function">pr_info</span><span class="token punctuation">(</span><span class="token string">&#34;dmabuf release\n&#34;</span><span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token function">put_page</span><span class="token punctuation">(</span>page<span class="token punctuation">)</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token keyword">static</span> <span class="token keyword">void</span> <span class="token operator">*</span><span class="token function">exporter_vmap</span><span class="token punctuation">(</span><span class="token keyword">struct</span> dma_buf <span class="token operator">*</span>dma_buf<span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token keyword">struct</span> page <span class="token operator">*</span>page <span class="token operator">&#61;</span> dma_buf<span class="token operator">-&gt;</span>priv<span class="token punctuation">;</span>

	<span class="token keyword">return</span> <span class="token function">vmap</span><span class="token punctuation">(</span><span class="token operator">&amp;</span>page<span class="token punctuation">,</span> <span class="token number">1</span><span class="token punctuation">,</span> <span class="token number">0</span><span class="token punctuation">,</span> PAGE_KERNEL<span class="token punctuation">)</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token keyword">static</span> <span class="token keyword">void</span> <span class="token function">exporter_vunmap</span><span class="token punctuation">(</span><span class="token keyword">struct</span> dma_buf <span class="token operator">*</span>dma_buf<span class="token punctuation">,</span> <span class="token keyword">void</span> <span class="token operator">*</span>vaddr<span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token function">vunmap</span><span class="token punctuation">(</span>vaddr<span class="token punctuation">)</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token keyword">static</span> <span class="token keyword">void</span> <span class="token operator">*</span><span class="token function">exporter_kmap_atomic</span><span class="token punctuation">(</span><span class="token keyword">struct</span> dma_buf <span class="token operator">*</span>dma_buf<span class="token punctuation">,</span> <span class="token keyword">unsigned</span> <span class="token keyword">long</span> page_num<span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token keyword">struct</span> page <span class="token operator">*</span>page <span class="token operator">&#61;</span> dma_buf<span class="token operator">-&gt;</span>priv<span class="token punctuation">;</span>

	<span class="token keyword">return</span> <span class="token function">kmap_atomic</span><span class="token punctuation">(</span>page<span class="token punctuation">)</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token keyword">static</span> <span class="token keyword">void</span> <span class="token function">exporter_kunmap_atomic</span><span class="token punctuation">(</span><span class="token keyword">struct</span> dma_buf <span class="token operator">*</span>dma_buf<span class="token punctuation">,</span> <span class="token keyword">unsigned</span> <span class="token keyword">long</span> page_num<span class="token punctuation">,</span> <span class="token keyword">void</span> <span class="token operator">*</span>addr<span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token function">kunmap_atomic</span><span class="token punctuation">(</span>addr<span class="token punctuation">)</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token keyword">static</span> <span class="token keyword">void</span> <span class="token operator">*</span><span class="token function">exporter_kmap</span><span class="token punctuation">(</span><span class="token keyword">struct</span> dma_buf <span class="token operator">*</span>dma_buf<span class="token punctuation">,</span> <span class="token keyword">unsigned</span> <span class="token keyword">long</span> page_num<span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token keyword">struct</span> page <span class="token operator">*</span>page <span class="token operator">&#61;</span> dma_buf<span class="token operator">-&gt;</span>priv<span class="token punctuation">;</span>

	<span class="token keyword">return</span> <span class="token function">kmap</span><span class="token punctuation">(</span>page<span class="token punctuation">)</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token keyword">static</span> <span class="token keyword">void</span> <span class="token function">exporter_kunmap</span><span class="token punctuation">(</span><span class="token keyword">struct</span> dma_buf <span class="token operator">*</span>dma_buf<span class="token punctuation">,</span> <span class="token keyword">unsigned</span> <span class="token keyword">long</span> page_num<span class="token punctuation">,</span> <span class="token keyword">void</span> <span class="token operator">*</span>addr<span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token keyword">struct</span> page <span class="token operator">*</span>page <span class="token operator">&#61;</span> dma_buf<span class="token operator">-&gt;</span>priv<span class="token punctuation">;</span>

	<span class="token keyword">return</span> <span class="token function">kunmap</span><span class="token punctuation">(</span>page<span class="token punctuation">)</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token keyword">static</span> <span class="token keyword">int</span> <span class="token function">exporter_mmap</span><span class="token punctuation">(</span><span class="token keyword">struct</span> dma_buf <span class="token operator">*</span>dma_buf<span class="token punctuation">,</span> <span class="token keyword">struct</span> vm_area_struct <span class="token operator">*</span>vma<span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token keyword">struct</span> page <span class="token operator">*</span>page <span class="token operator">&#61;</span> dma_buf<span class="token operator">-&gt;</span>priv<span class="token punctuation">;</span>

	<span class="token keyword">return</span> <span class="token function">remap_pfn_range</span><span class="token punctuation">(</span>vma<span class="token punctuation">,</span> vma<span class="token operator">-&gt;</span>vm_start<span class="token punctuation">,</span> <span class="token function">page_to_pfn</span><span class="token punctuation">(</span>page<span class="token punctuation">)</span><span class="token punctuation">,</span>
				PAGE_SIZE<span class="token punctuation">,</span> vma<span class="token operator">-&gt;</span>vm_page_prot<span class="token punctuation">)</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token keyword">static</span> <span class="token keyword">int</span> <span class="token function">exporter_begin_cpu_access</span><span class="token punctuation">(</span><span class="token keyword">struct</span> dma_buf <span class="token operator">*</span>dmabuf<span class="token punctuation">,</span>
				      <span class="token keyword">enum</span> dma_data_direction dir<span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token keyword">struct</span> dma_buf_attachment <span class="token operator">*</span>attachment<span class="token punctuation">;</span>
	<span class="token keyword">struct</span> sg_table <span class="token operator">*</span>table<span class="token punctuation">;</span>

	<span class="token keyword">if</span> <span class="token punctuation">(</span><span class="token function">list_empty</span><span class="token punctuation">(</span><span class="token operator">&amp;</span>dmabuf<span class="token operator">-&gt;</span>attachments<span class="token punctuation">)</span><span class="token punctuation">)</span>
		<span class="token keyword">return</span> <span class="token number">0</span><span class="token punctuation">;</span>

	attachment <span class="token operator">&#61;</span> <span class="token function">list_first_entry</span><span class="token punctuation">(</span><span class="token operator">&amp;</span>dmabuf<span class="token operator">-&gt;</span>attachments<span class="token punctuation">,</span> <span class="token keyword">struct</span> dma_buf_attachment<span class="token punctuation">,</span> node<span class="token punctuation">)</span><span class="token punctuation">;</span>
	table <span class="token operator">&#61;</span> attachment<span class="token operator">-&gt;</span>priv<span class="token punctuation">;</span>
	<span class="token function">dma_sync_sg_for_cpu</span><span class="token punctuation">(</span><span class="token constant">NULL</span><span class="token punctuation">,</span> table<span class="token operator">-&gt;</span>sgl<span class="token punctuation">,</span> <span class="token number">1</span><span class="token punctuation">,</span> dir<span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token keyword">return</span> <span class="token number">0</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token keyword">static</span> <span class="token keyword">int</span> <span class="token function">exporter_end_cpu_access</span><span class="token punctuation">(</span><span class="token keyword">struct</span> dma_buf <span class="token operator">*</span>dmabuf<span class="token punctuation">,</span>
				<span class="token keyword">enum</span> dma_data_direction dir<span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token keyword">struct</span> dma_buf_attachment <span class="token operator">*</span>attachment<span class="token punctuation">;</span>
	<span class="token keyword">struct</span> sg_table <span class="token operator">*</span>table<span class="token punctuation">;</span>

	<span class="token keyword">if</span> <span class="token punctuation">(</span><span class="token function">list_empty</span><span class="token punctuation">(</span><span class="token operator">&amp;</span>dmabuf<span class="token operator">-&gt;</span>attachments<span class="token punctuation">)</span><span class="token punctuation">)</span>
		<span class="token keyword">return</span> <span class="token number">0</span><span class="token punctuation">;</span>

	attachment <span class="token operator">&#61;</span> <span class="token function">list_first_entry</span><span class="token punctuation">(</span><span class="token operator">&amp;</span>dmabuf<span class="token operator">-&gt;</span>attachments<span class="token punctuation">,</span> <span class="token keyword">struct</span> dma_buf_attachment<span class="token punctuation">,</span> node<span class="token punctuation">)</span><span class="token punctuation">;</span>
	table <span class="token operator">&#61;</span> attachment<span class="token operator">-&gt;</span>priv<span class="token punctuation">;</span>
	<span class="token function">dma_sync_sg_for_device</span><span class="token punctuation">(</span><span class="token constant">NULL</span><span class="token punctuation">,</span> table<span class="token operator">-&gt;</span>sgl<span class="token punctuation">,</span> <span class="token number">1</span><span class="token punctuation">,</span> dir<span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token keyword">return</span> <span class="token number">0</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token keyword">static</span> <span class="token keyword">const</span> <span class="token keyword">struct</span> dma_buf_ops exp_dmabuf_ops <span class="token operator">&#61;</span> <span class="token punctuation">{<!-- --></span>
	<span class="token punctuation">.</span>attach <span class="token operator">&#61;</span> exporter_attach<span class="token punctuation">,</span>
	<span class="token punctuation">.</span>detach <span class="token operator">&#61;</span> exporter_detach<span class="token punctuation">,</span>
	<span class="token punctuation">.</span>map_dma_buf <span class="token operator">&#61;</span> exporter_map_dma_buf<span class="token punctuation">,</span>
	<span class="token punctuation">.</span>unmap_dma_buf <span class="token operator">&#61;</span> exporter_unmap_dma_buf<span class="token punctuation">,</span>
	<span class="token punctuation">.</span>release <span class="token operator">&#61;</span> exporter_release<span class="token punctuation">,</span>
	<span class="token punctuation">.</span>map <span class="token operator">&#61;</span> exporter_kmap<span class="token punctuation">,</span>
	<span class="token punctuation">.</span>unmap <span class="token operator">&#61;</span> exporter_kunmap<span class="token punctuation">,</span>
	<span class="token punctuation">.</span>map_atomic <span class="token operator">&#61;</span> exporter_kmap_atomic<span class="token punctuation">,</span>
	<span class="token punctuation">.</span>unmap_atomic <span class="token operator">&#61;</span> exporter_kunmap_atomic<span class="token punctuation">,</span>
	<span class="token punctuation">.</span>mmap <span class="token operator">&#61;</span> exporter_mmap<span class="token punctuation">,</span>
	<span class="token punctuation">.</span>vmap <span class="token operator">&#61;</span> exporter_vmap<span class="token punctuation">,</span>
	<span class="token punctuation">.</span>vunmap <span class="token operator">&#61;</span> exporter_vunmap<span class="token punctuation">,</span>
	<span class="token punctuation">.</span>begin_cpu_access <span class="token operator">&#61;</span> exporter_begin_cpu_access<span class="token punctuation">,</span>
	<span class="token punctuation">.</span>end_cpu_access <span class="token operator">&#61;</span> exporter_end_cpu_access<span class="token punctuation">,</span>
<span class="token punctuation">}</span><span class="token punctuation">;</span>

<span class="token keyword">static</span> <span class="token keyword">struct</span> dma_buf <span class="token operator">*</span><span class="token function">exporter_alloc_page</span><span class="token punctuation">(</span><span class="token keyword">void</span><span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token function">DEFINE_DMA_BUF_EXPORT_INFO</span><span class="token punctuation">(</span>exp_info<span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token keyword">struct</span> dma_buf <span class="token operator">*</span>dmabuf<span class="token punctuation">;</span>
	<span class="token keyword">struct</span> page <span class="token operator">*</span>page<span class="token punctuation">;</span>

	page <span class="token operator">&#61;</span> <span class="token function">alloc_page</span><span class="token punctuation">(</span>GFP_KERNEL<span class="token punctuation">)</span><span class="token punctuation">;</span>

	exp_info<span class="token punctuation">.</span>ops <span class="token operator">&#61;</span> <span class="token operator">&amp;</span>exp_dmabuf_ops<span class="token punctuation">;</span>
	exp_info<span class="token punctuation">.</span>size <span class="token operator">&#61;</span> PAGE_SIZE<span class="token punctuation">;</span>
	exp_info<span class="token punctuation">.</span>flags <span class="token operator">&#61;</span> O_CLOEXEC<span class="token punctuation">;</span>
	exp_info<span class="token punctuation">.</span>priv <span class="token operator">&#61;</span> page<span class="token punctuation">;</span>

	dmabuf <span class="token operator">&#61;</span> <span class="token function">dma_buf_export</span><span class="token punctuation">(</span><span class="token operator">&amp;</span>exp_info<span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token function">sprintf</span><span class="token punctuation">(</span><span class="token function">page_address</span><span class="token punctuation">(</span>page<span class="token punctuation">)</span><span class="token punctuation">,</span> <span class="token string">&#34;hello world!&#34;</span><span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token keyword">return</span> dmabuf<span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token keyword">static</span> <span class="token keyword">long</span> <span class="token function">exporter_ioctl</span><span class="token punctuation">(</span><span class="token keyword">struct</span> file <span class="token operator">*</span>filp<span class="token punctuation">,</span> <span class="token keyword">unsigned</span> <span class="token keyword">int</span> cmd<span class="token punctuation">,</span> <span class="token keyword">unsigned</span> <span class="token keyword">long</span> arg<span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token keyword">int</span> fd <span class="token operator">&#61;</span> <span class="token function">dma_buf_fd</span><span class="token punctuation">(</span>dmabuf_exported<span class="token punctuation">,</span> O_CLOEXEC<span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token keyword">if</span> <span class="token punctuation">(</span><span class="token function">copy_to_user</span><span class="token punctuation">(</span><span class="token punctuation">(</span><span class="token keyword">int</span> __user <span class="token operator">*</span><span class="token punctuation">)</span>arg<span class="token punctuation">,</span> <span class="token operator">&amp;</span>fd<span class="token punctuation">,</span> <span class="token keyword">sizeof</span><span class="token punctuation">(</span>fd<span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
		<span class="token keyword">return</span> <span class="token operator">-</span>EFAULT<span class="token punctuation">;</span>

	<span class="token keyword">return</span> <span class="token number">0</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>
 
<span class="token keyword">static</span> <span class="token keyword">struct</span> file_operations exporter_fops <span class="token operator">&#61;</span> <span class="token punctuation">{<!-- --></span>
	<span class="token punctuation">.</span>owner   <span class="token operator">&#61;</span> THIS_MODULE<span class="token punctuation">,</span>
	<span class="token punctuation">.</span>unlocked_ioctl   <span class="token operator">&#61;</span> exporter_ioctl<span class="token punctuation">,</span>
<span class="token punctuation">}</span><span class="token punctuation">;</span>
 
<span class="token keyword">static</span> <span class="token keyword">struct</span> miscdevice mdev <span class="token operator">&#61;</span> <span class="token punctuation">{<!-- --></span>
	<span class="token punctuation">.</span>minor <span class="token operator">&#61;</span> MISC_DYNAMIC_MINOR<span class="token punctuation">,</span>
	<span class="token punctuation">.</span>name <span class="token operator">&#61;</span> <span class="token string">&#34;exporter&#34;</span><span class="token punctuation">,</span>
	<span class="token punctuation">.</span>fops <span class="token operator">&#61;</span> <span class="token operator">&amp;</span>exporter_fops<span class="token punctuation">,</span>
<span class="token punctuation">}</span><span class="token punctuation">;</span>
 
<span class="token keyword">static</span> <span class="token keyword">int</span> __init <span class="token function">exporter_init</span><span class="token punctuation">(</span><span class="token keyword">void</span><span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	dmabuf_exported <span class="token operator">&#61;</span> <span class="token function">exporter_alloc_page</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token keyword">return</span> <span class="token function">misc_register</span><span class="token punctuation">(</span><span class="token operator">&amp;</span>mdev<span class="token punctuation">)</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token keyword">static</span> <span class="token keyword">void</span> __exit <span class="token function">exporter_exit</span><span class="token punctuation">(</span><span class="token keyword">void</span><span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token function">misc_deregister</span><span class="token punctuation">(</span><span class="token operator">&amp;</span>mdev<span class="token punctuation">)</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token function">module_init</span><span class="token punctuation">(</span>exporter_init<span class="token punctuation">)</span><span class="token punctuation">;</span>
<span class="token function">module_exit</span><span class="token punctuation">(</span>exporter_exit<span class="token punctuation">)</span><span class="token punctuation">;</span>

</code></pre> 
<p><strong>importer 驱动</strong><br /> 将前几篇的 <a href="https://blog.csdn.net/hexiaolong2009/article/details/102596761#t4">importer-kmap.c</a> 和 <a href="https://blog.csdn.net/hexiaolong2009/article/details/102596772#t7">importer-sg.c</a> 合二为一&#xff0c;如下&#xff1a;</p> 
<p><a href="https://github.com/hexiaolong2008/sample-code/blob/master/dma-buf/08/importer-page.c">importer-page.c</a></p> 
<pre><code class="prism language-c"><span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/dma-buf.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/module.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/miscdevice.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/slab.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/uaccess.h&gt;</span></span>

<span class="token keyword">static</span> <span class="token keyword">int</span> <span class="token function">importer_test1</span><span class="token punctuation">(</span><span class="token keyword">struct</span> dma_buf <span class="token operator">*</span>dmabuf<span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token keyword">void</span> <span class="token operator">*</span>vaddr<span class="token punctuation">;</span>

	vaddr <span class="token operator">&#61;</span> <span class="token function">dma_buf_kmap</span><span class="token punctuation">(</span>dmabuf<span class="token punctuation">,</span> <span class="token number">0</span><span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token function">pr_info</span><span class="token punctuation">(</span><span class="token string">&#34;read from dmabuf kmap: %s\n&#34;</span><span class="token punctuation">,</span> <span class="token punctuation">(</span><span class="token keyword">char</span> <span class="token operator">*</span><span class="token punctuation">)</span>vaddr<span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token function">dma_buf_kunmap</span><span class="token punctuation">(</span>dmabuf<span class="token punctuation">,</span> <span class="token number">0</span><span class="token punctuation">,</span> vaddr<span class="token punctuation">)</span><span class="token punctuation">;</span>

	vaddr <span class="token operator">&#61;</span> <span class="token function">dma_buf_vmap</span><span class="token punctuation">(</span>dmabuf<span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token function">pr_info</span><span class="token punctuation">(</span><span class="token string">&#34;read from dmabuf vmap: %s\n&#34;</span><span class="token punctuation">,</span> <span class="token punctuation">(</span><span class="token keyword">char</span> <span class="token operator">*</span><span class="token punctuation">)</span>vaddr<span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token function">dma_buf_vunmap</span><span class="token punctuation">(</span>dmabuf<span class="token punctuation">,</span> vaddr<span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token keyword">return</span> <span class="token number">0</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token keyword">static</span> <span class="token keyword">int</span> <span class="token function">importer_test2</span><span class="token punctuation">(</span><span class="token keyword">struct</span> dma_buf <span class="token operator">*</span>dmabuf<span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token keyword">struct</span> dma_buf_attachment <span class="token operator">*</span>attachment<span class="token punctuation">;</span>
	<span class="token keyword">struct</span> sg_table <span class="token operator">*</span>table<span class="token punctuation">;</span>
	<span class="token keyword">struct</span> device <span class="token operator">*</span>dev<span class="token punctuation">;</span>
	<span class="token keyword">unsigned</span> <span class="token keyword">int</span> reg_addr<span class="token punctuation">,</span> reg_size<span class="token punctuation">;</span>

	dev <span class="token operator">&#61;</span> <span class="token function">kzalloc</span><span class="token punctuation">(</span><span class="token keyword">sizeof</span><span class="token punctuation">(</span><span class="token operator">*</span>dev<span class="token punctuation">)</span><span class="token punctuation">,</span> GFP_KERNEL<span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token function">dev_set_name</span><span class="token punctuation">(</span>dev<span class="token punctuation">,</span> <span class="token string">&#34;importer&#34;</span><span class="token punctuation">)</span><span class="token punctuation">;</span>

	attachment <span class="token operator">&#61;</span> <span class="token function">dma_buf_attach</span><span class="token punctuation">(</span>dmabuf<span class="token punctuation">,</span> dev<span class="token punctuation">)</span><span class="token punctuation">;</span>
	table <span class="token operator">&#61;</span> <span class="token function">dma_buf_map_attachment</span><span class="token punctuation">(</span>attachment<span class="token punctuation">,</span> DMA_BIDIRECTIONAL<span class="token punctuation">)</span><span class="token punctuation">;</span>

	reg_addr <span class="token operator">&#61;</span> <span class="token function">sg_dma_address</span><span class="token punctuation">(</span>table<span class="token operator">-&gt;</span>sgl<span class="token punctuation">)</span><span class="token punctuation">;</span>
	reg_size <span class="token operator">&#61;</span> <span class="token function">sg_dma_len</span><span class="token punctuation">(</span>table<span class="token operator">-&gt;</span>sgl<span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token function">pr_info</span><span class="token punctuation">(</span><span class="token string">&#34;reg_addr &#61; 0x%08x, reg_size &#61; 0x%08x\n&#34;</span><span class="token punctuation">,</span> reg_addr<span class="token punctuation">,</span> reg_size<span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token function">dma_buf_unmap_attachment</span><span class="token punctuation">(</span>attachment<span class="token punctuation">,</span> table<span class="token punctuation">,</span> DMA_BIDIRECTIONAL<span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token function">dma_buf_detach</span><span class="token punctuation">(</span>dmabuf<span class="token punctuation">,</span> attachment<span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token keyword">return</span> <span class="token number">0</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token keyword">static</span> <span class="token keyword">long</span> <span class="token function">importer_ioctl</span><span class="token punctuation">(</span><span class="token keyword">struct</span> file <span class="token operator">*</span>filp<span class="token punctuation">,</span> <span class="token keyword">unsigned</span> <span class="token keyword">int</span> cmd<span class="token punctuation">,</span> <span class="token keyword">unsigned</span> <span class="token keyword">long</span> arg<span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token keyword">int</span> fd<span class="token punctuation">;</span>
	<span class="token keyword">struct</span> dma_buf <span class="token operator">*</span>dmabuf<span class="token punctuation">;</span>

	<span class="token keyword">if</span> <span class="token punctuation">(</span><span class="token function">copy_from_user</span><span class="token punctuation">(</span><span class="token operator">&amp;</span>fd<span class="token punctuation">,</span> <span class="token punctuation">(</span><span class="token keyword">void</span> __user <span class="token operator">*</span><span class="token punctuation">)</span>arg<span class="token punctuation">,</span> <span class="token keyword">sizeof</span><span class="token punctuation">(</span><span class="token keyword">int</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
		<span class="token keyword">return</span> <span class="token operator">-</span>EFAULT<span class="token punctuation">;</span>

	dmabuf <span class="token operator">&#61;</span> <span class="token function">dma_buf_get</span><span class="token punctuation">(</span>fd<span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token function">importer_test1</span><span class="token punctuation">(</span>dmabuf<span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token function">importer_test2</span><span class="token punctuation">(</span>dmabuf<span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token function">dma_buf_put</span><span class="token punctuation">(</span>dmabuf<span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token keyword">return</span> <span class="token number">0</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>
 
<span class="token keyword">static</span> <span class="token keyword">struct</span> file_operations importer_fops <span class="token operator">&#61;</span> <span class="token punctuation">{<!-- --></span>
	<span class="token punctuation">.</span>owner	<span class="token operator">&#61;</span> THIS_MODULE<span class="token punctuation">,</span>
	<span class="token punctuation">.</span>unlocked_ioctl	<span class="token operator">&#61;</span> importer_ioctl<span class="token punctuation">,</span>
<span class="token punctuation">}</span><span class="token punctuation">;</span>
 
<span class="token keyword">static</span> <span class="token keyword">struct</span> miscdevice mdev <span class="token operator">&#61;</span> <span class="token punctuation">{<!-- --></span>
	<span class="token punctuation">.</span>minor <span class="token operator">&#61;</span> MISC_DYNAMIC_MINOR<span class="token punctuation">,</span>
	<span class="token punctuation">.</span>name <span class="token operator">&#61;</span> <span class="token string">&#34;importer&#34;</span><span class="token punctuation">,</span>
	<span class="token punctuation">.</span>fops <span class="token operator">&#61;</span> <span class="token operator">&amp;</span>importer_fops<span class="token punctuation">,</span>
<span class="token punctuation">}</span><span class="token punctuation">;</span>
 
<span class="token keyword">static</span> <span class="token keyword">int</span> __init <span class="token function">importer_init</span><span class="token punctuation">(</span><span class="token keyword">void</span><span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token keyword">return</span> <span class="token function">misc_register</span><span class="token punctuation">(</span><span class="token operator">&amp;</span>mdev<span class="token punctuation">)</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token keyword">static</span> <span class="token keyword">void</span> __exit <span class="token function">importer_exit</span><span class="token punctuation">(</span><span class="token keyword">void</span><span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token function">misc_deregister</span><span class="token punctuation">(</span><span class="token operator">&amp;</span>mdev<span class="token punctuation">)</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token function">module_init</span><span class="token punctuation">(</span>importer_init<span class="token punctuation">)</span><span class="token punctuation">;</span>
<span class="token function">module_exit</span><span class="token punctuation">(</span>importer_exit<span class="token punctuation">)</span><span class="token punctuation">;</span>
</code></pre> 
<h3><a id="_331"></a>运行</h3> 
<p>在 my-qemu 仿真环境中执行如下命令&#xff1a;</p> 
<pre><code class="prism language-handlebars"><span class="token punctuation">#</span> <span class="token variable">insmod</span> <span class="token punctuation">/</span><span class="token variable">lib</span><span class="token punctuation">/</span><span class="token variable">modules</span><span class="token punctuation">/</span><span class="token number">4.14</span><span class="token number">.143</span><span class="token block keyword">/kernel/drivers/dma-buf/exporter-page.ko</span>
<span class="token punctuation">#</span> <span class="token variable">insmod</span> <span class="token punctuation">/</span><span class="token variable">lib</span><span class="token punctuation">/</span><span class="token variable">modules</span><span class="token punctuation">/</span><span class="token number">4.14</span><span class="token number">.143</span><span class="token block keyword">/kernel/drivers/dma-buf/importer-page.ko</span>
</code></pre> 
<p>将看到如下打印结果&#xff1a;</p> 
<pre><code>read from dmabuf kmap: hello world!
read from dmabuf vmap: hello world!
dmabuf attach device: importer
reg_addr &#61; 0x7f6ee000, reg_size &#61; 0x00001000
dmabuf detach device: importer
</code></pre> 
<h3><a id="_346"></a>资源</h3> 
<table><thead><tr><th align="left"></th><th align="left"></th></tr></thead><tbody><tr><td align="left">内核源码</td><td align="left"><a href="https://mirrors.edge.kernel.org/pub/linux/kernel/v4.x/linux-4.14.143.tar.xz" rel="nofollow">4.14.143</a></td></tr><tr><td align="left">示例源码</td><td align="left"><a href="https://github.com/hexiaolong2008/sample-code/tree/master/dma-buf/08">hexiaolong2008-GitHub/sample-code/dma-buf/08</a></td></tr><tr><td align="left">开发平台</td><td align="left">Ubuntu14.04/16.04</td></tr><tr><td align="left">运行平台</td><td align="left"><a href="https://github.com/hexiaolong2008/my-qemu">my-qemu 仿真环境</a></td></tr></tbody></table>
<h3><a id="_354"></a>结语</h3> 
<p>其实本篇的真实目的是为下篇<a href="https://blog.csdn.net/hexiaolong2009/article/details/103795381">《dma-buf 由浅入深 —— ION 简化版》</a>做铺垫的&#xff0c;通过本篇能够对 page 的相关操作有个印象&#xff0c;这样才方便下一篇的理解。好了&#xff0c;我们赶紧去看下一篇吧&#xff01;</p> 
<br /> 
<br /> 
<br /> 
<p>上一篇&#xff1a;<a href="https://blog.csdn.net/hexiaolong2009/article/details/102596825">《dma-buf 由浅入深&#xff08;六&#xff09; —— begin / end cpu_access》</a><br /> 下一篇&#xff1a;<a href="https://blog.csdn.net/hexiaolong2009/article/details/103795381">《dma-buf 由浅入深&#xff08;八&#xff09;—— ION 简化版》</a><br /> 文章汇总&#xff1a;<a href="https://blog.csdn.net/hexiaolong2009/article/details/83720940">《DRM&#xff08;Direct Rendering Manager&#xff09;学习简介》</a></p>
                </div>
                <link href="https://csdnimg.cn/release/blogv2/dist/mdeditor/css/editerView/markdown_views-375c595788.css" rel="stylesheet">
                <link href="https://csdnimg.cn/release/blogv2/dist/mdeditor/css/style-e504d6a974.css" rel="stylesheet">
        </div>
    </article>

  <script>
    $(function() {
      setTimeout(function () {
        var mathcodeList = document.querySelectorAll('.htmledit_views img.mathcode');
        if (mathcodeList.length > 0) {
          for (let i = 0; i < mathcodeList.length; i++) {
            if (mathcodeList[i].complete) {
              if (mathcodeList[i].naturalWidth === 0 || mathcodeList[i].naturalHeight === 0) {
                var alt = mathcodeList[i].alt;
                alt = '\\(' + alt + '\\)';
                var curSpan = $('<span class="img-codecogs"></span>');
                curSpan.text(alt);
                $(mathcodeList[i]).before(curSpan);
                $(mathcodeList[i]).remove();
              }
            } else {
              mathcodeList[i].onerror = function() {
                var alt = mathcodeList[i].alt;
                alt = '\\(' + alt + '\\)';
                var curSpan = $('<span class="img-codecogs"></span>');
                curSpan.text(alt);
                $(mathcodeList[i]).before(curSpan);
                $(mathcodeList[i]).remove();
              };
            }
          }
          MathJax.Hub.Queue(["Typeset",MathJax.Hub]);
        }
      }, 500)
    });
  </script>
</div>
<div class="directory-boxshadow-dialog" style="display:none;">
  <div class="directory-boxshadow-dialog-box">
  </div>
   <div class="vip-limited-time-offer-box-new" id="vip-limited-time-offer-box-new">
      <img class="limited-img limited-img-new" src="https://csdnimg.cn/release/blogv2/dist/pc/img/vip-limited-close-newWhite.png">
      <div class="vip-limited-time-top">
        确定要放弃本次机会？
      </div>
      <span class="vip-limited-time-text">福利倒计时</span>
      <div class="limited-time-box-new">
        <span class="time-hour"></span>
        <i>:</i>
        <span class="time-minite"></span>
        <i>:</i>
        <span class="time-second"></span>
      </div>
      <div class="limited-time-vip-box">
        <p>
          <img class="coupon-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/vip-limited-close-roup.png">
          <span class="def">立减 ¥</span>
          <span class="active limited-num"></span>
        </p>
        <span class="">普通VIP年卡可用</span>
      </div>
      <a class="limited-time-btn-new" href="https://mall.csdn.net/vip" data-report-click='{"spm":"1001.2101.3001.9621"}' data-report-query='spm=1001.2101.3001.9621'>立即使用</a>
  </div>
</div>
    <div class="more-toolbox-new more-toolbar" id="toolBarBox">
      <div class="left-toolbox">
        <div class="toolbox-left">
            <div class="profile-box">
              <a class="profile-href" target="_blank" href="https://blog.csdn.net/hexiaolong2009"><img class="profile-img" src="https://profile-avatar.csdnimg.cn/13e0f6961a114b5cb09ff65ed930a667_hexiaolong2009.jpg!1">
                <span class="profile-name">
                  何小龙
                </span>
              </a>
            </div>
            <div class="profile-attend">
                <a class="tool-attend tool-bt-button tool-bt-attend" href="javascript:;" data-report-view='{"mod":"1592215036_002","spm":"1001.2101.3001.4232","extend1":"关注"}'>关注</a>
              <a class="tool-item-follow active-animation" style="display:none;">关注</a>
            </div>
        </div>
        <div class="toolbox-middle">
          <ul class="toolbox-list">
            <li class="tool-item tool-item-size tool-active is-like" id="is-like" data-type="bottom">
              <a class="tool-item-href">
                <img style="display:none;" id="is-like-imgactive-animation-like" class="animation-dom active-animation" src="https://csdnimg.cn/release/blogv2/dist/pc/img/tobarThumbUpactive.png" alt="">
                <img class="isactive" style="display:none" id="is-like-imgactive" src="https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/like-active.png" alt="">
                <img class="isdefault" style="display:block" id="is-like-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/like.png" alt="">
                <span id="spanCount" class="count ">
                    13
                </span>
              </a>
              <div class="tool-hover-tip"><span class="text space">点赞</span></div>
            </li>
            <li class="tool-item tool-item-size tool-active is-unlike" id="is-unlike">
              <a class="tool-item-href">
                <img class="isactive" style="margin-right:0px;display:none" id="is-unlike-imgactive" src="https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/unlike-active.png" alt="">
                <img class="isdefault" style="margin-right:0px;display:block" id="is-unlike-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/unlike.png" alt="">
                <span id="unlikeCount" class="count "></span>
              </a>
              <div class="tool-hover-tip"><span class="text space">踩</span></div>
            </li>
            <li class="tool-item tool-item-size tool-active is-collection ">
              <a class="tool-item-href" href="javascript:;" data-report-click='{"mod":"popu_824","spm":"1001.2101.3001.4130","ab":"new"}'>
                <img style="display:none" id="is-collection-img-collection" class="animation-dom active-animation" src="https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/collect-active.png" alt="">
                <img class="isdefault" id="is-collection-img" style="display:block" src="https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/collect.png" alt="">
                <img class="isactive" id="is-collection-imgactive" style="display:none" src="https://csdnimg.cn/release/blogv2/dist/pc/img/newCollectActive.png" alt="">
                <span class="count get-collection " data-num="15" id="get-collection">
                    15
                </span>
              </a>
              <div class="tool-hover-tip collect">
                <div class="collect-operate-box">
                  <span class="collect-text" id="is-collection">
                    收藏
                  </span>
                </div>
              </div>
              <div class="tool-active-list">
                <div class="text">
                  觉得还不错?
                  <span class="collect-text" id="tool-active-list-collection">
                    一键收藏
                  </span>
                 <img id="tool-active-list-close" src="https://csdnimg.cn/release/blogv2/dist/pc/img/collectionCloseWhite.png" alt="">
                </div>
              </div>
            </li>
            <li class="tool-item tool-item-size tool-active tool-item-comment">
              <div class="guide-rr-first">
                <img src="https://csdnimg.cn/release/blogv2/dist/pc/img/guideRedReward01.png" alt="">
                <button class="btn-guide-known">知道了</button>
              </div>
                <a class="tool-item-href go-side-comment" data-report-click='{"spm":"1001.2101.3001.7009"}'>
                <img class="isdefault" src="https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/comment.png" alt="">
                <span class="count">
                    6
                </span>
              </a>
              <div class="tool-hover-tip"><span class="text space">评论</span></div>
            </li>
            <li class="tool-item tool-item-size tool-active tool-QRcode" data-type="article" id="tool-share">
              <a class="tool-item-href" href="javascript:;" data-report-view='{"spm":"3001.4129","extra":{"type":"blogdetail"}}'>
                <img class="isdefault" src="https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/share.png" alt="">
                <span class="count">分享</span>
              </a>
                <div class="QRcode" id="tool-QRcode">
                <div class="share-bg-box">
                  <div class="share-content">
                    <a id="copyPosterUrl" data-type="link" class="btn-share">复制链接</a>
                  </div>
                  <div class="share-content">
                    <a class="btn-share" data-type="qq">分享到 QQ</a>
                  </div>
                  <div class="share-content">
                    <a class="btn-share" data-type="weibo">分享到新浪微博</a>
                  </div>
                  <div class="share-code">
                    <div class="share-code-box" id='shareCode'></div>
                    <div class="share-code-text">
                      <img src="https://csdnimg.cn/release/blogv2/dist/pc/img/share/icon-wechat.png" alt="">扫一扫
                    </div>
                  </div>
                </div>
              </div>
            </li>
                <li class="tool-item tool-item-size tool-active tool-item-reward">
                  <a class="tool-item-href" href="javascript:;" data-report-click='{"mod":"popu_830","spm":"1001.2101.3001.4237","dest":"","ab":"new"}'>
                    <img class="isdefault reward-bt" id="rewardBtNew" src="https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/reward.png" alt="打赏">
                    <span class="count">打赏</span>
                  </a>
                  <div class="tool-hover-tip"><span class="text space">打赏</span></div>
                </li>
          <li class="tool-item tool-item-size tool-active is-more" id="is-more">
            <a class="tool-item-href">
              <img class="isdefault" style="margin-right:0px;display:block" src="https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/more.png" alt="">
              <span class="count"></span>
            </a>
            <div class="more-opt-box">
              <div class="mini-box">
                    <a class="tool-item-href" id="rewardBtNewHide" data-report-click='{"spm":"3001.4237","extra":"{\"type\":\"hide\"}"}'>
                      <img class="isdefault reward-bt" src="https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/reward.png" alt="打赏">
                      <span class="count">打赏</span>
                    </a>
                <a class="tool-item-href" id="toolReportBtnHide">
                  <img class="isdefault" src="https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/report.png" alt="">
                  <span class="count">举报</span>
                </a>
              </div>
              <div class="normal-box">
                <a class="tool-item-href" id="toolReportBtnHideNormal">
                  <img class="isdefault" src="https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/report.png" alt="">
                  <span class="count">举报</span>
                </a>
              </div>
            </div>
          </li>
        </ul>
      </div>
      <div class="toolbox-right">
            <div class="tool-directory">
                <a class="bt-columnlist-show"
                  data-id="9281458"
                  data-free="true"
                  data-description="分享本人学习Linux DRM (Direct Rendering Manager) 图形架构的经验总结，并以最简单的示例展示如何编写DRM应用程序和驱动程序，简单易懂，适合初学者。"
                  data-subscribe="false"
                  data-title="DRM (Direct Rendering Manager)"
                  data-img="https://i-blog.csdnimg.cn/blog_column_migrate/c03d60f8b24740d58f94a787024155be.png?x-oss-process=image/resize,m_fixed,h_224,w_224"
                  data-url="https://blog.csdn.net/hexiaolong2009/category_9281458.html"
                  data-sum="29"
                  data-people="1235"
                  data-price="0"
                  data-hotRank="0"
                  data-status="true"
                  data-oldprice="0"
                  data-join="false"
                  data-studyvip="false"
                  data-studysubscribe="false"
                  data-report-view='{"spm":"1001.2101.3001.6334","extend1":"专栏目录"}'
                  data-report-click='{"spm":"1001.2101.3001.6334","extend1":"专栏目录"}'>专栏目录</a>
          </div>
</div>
</div>
</div>
<script type=text/javascript crossorigin src="https://csdnimg.cn/release/phoenix/production/qrcode-7c90a92189.min.js"></script>
<script type="text/javascript" crossorigin src="https://g.csdnimg.cn/common/csdn-login-box/csdn-login-box.js"></script>
<script type="text/javascript" crossorigin src="https://g.csdnimg.cn/collection-box/2.1.2/collection-box.js"></script>
                <div class="first-recommend-box recommend-box ">
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/weixin_42136255/article/details/133749854"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6661.1","mod":"popu_871","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant_t0.none-task-blog-2~default~BlogCommendFromBaidu~PaidSort-1-133749854-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"1","strategy":"2~default~BlogCommendFromBaidu~PaidSort","dest":"https://blog.csdn.net/weixin_42136255/article/details/133749854"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/weixin_42136255/article/details/133749854" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6661.1","mod":"popu_871","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant_t0.none-task-blog-2~default~BlogCommendFromBaidu~PaidSort-1-133749854-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"1","strategy":"2~default~BlogCommendFromBaidu~PaidSort","dest":"https://blog.csdn.net/weixin_42136255/article/details/133749854"}'  data-report-query='spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7EPaidSort-1-133749854-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7EPaidSort-1-133749854-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">How to transition ION to <em>DMA</em><em>-</em><em>BUF</em> heaps</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/weixin_42136255" target="_blank"><span class="blog-title">weixin_42136255的博客</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">10-10</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					1733
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/weixin_42136255/article/details/133749854" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6661.1","mod":"popu_871","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant_t0.none-task-blog-2~default~BlogCommendFromBaidu~PaidSort-1-133749854-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"1","strategy":"2~default~BlogCommendFromBaidu~PaidSort","dest":"https://blog.csdn.net/weixin_42136255/article/details/133749854"}'  data-report-query='spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7EPaidSort-1-133749854-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7EPaidSort-1-133749854-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">How to transition ION to <em>DMA</em><em>-</em><em>BUF</em> heaps</div>
			</a>
		</div>
	</div>
</div>
                </div>
            <script src="https://csdnimg.cn/release/blogv2/dist/components/js/pc_wap_commontools-829a4838ae.min.js" type="text/javascript" async></script>
              <div class="second-recommend-box recommend-box ">
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/isoftstone_HOS/article/details/127731103"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.1","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~PaidSort-1-127731103-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"1","strategy":"2~default~BlogCommendFromBaidu~PaidSort","dest":"https://blog.csdn.net/isoftstone_HOS/article/details/127731103"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/isoftstone_HOS/article/details/127731103" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.1","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~PaidSort-1-127731103-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"1","strategy":"2~default~BlogCommendFromBaidu~PaidSort","dest":"https://blog.csdn.net/isoftstone_HOS/article/details/127731103"}'  data-report-query='spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7EPaidSort-1-127731103-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7EPaidSort-1-127731103-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">RK系列开发板音频驱动适配指南<em>(</em>二<em>)</em></div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/isoftstone_HOS" target="_blank"><span class="blog-title">isoftstone_HOS的博客</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">11-07</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					1332
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/isoftstone_HOS/article/details/127731103" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.1","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~PaidSort-1-127731103-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"1","strategy":"2~default~BlogCommendFromBaidu~PaidSort","dest":"https://blog.csdn.net/isoftstone_HOS/article/details/127731103"}'  data-report-query='spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7EPaidSort-1-127731103-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7EPaidSort-1-127731103-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">上一篇文章RK系列开发板音频驱动适配指南<em>-</em>DAI模块适配中已经阐述音频驱动适配的DAI模块适配步骤以及核心代码的展示，本次主要介绍音频驱动适配中的<em>DMA</em>模块适配。</div>
			</a>
		</div>
	</div>
</div>
              </div>
<a id="commentBox" name="commentBox"></a>
  <div id="pcCommentBox" class="comment-box comment-box-new2 unlogin-comment-box-new" style="display:none">
      <div class="unlogin-comment-model">
          <span class="unlogin-comment-tit">6&nbsp;条评论</span>
        <span class="unlogin-comment-text">您还未登录，请先</span>
        <span class="unlogin-comment-bt">登录</span>
        <span class="unlogin-comment-text">后发表或查看评论</span>
      </div>
  </div>
  <div class="blog-comment-box-new" style="display: none;">
        <h1>6 条评论</h1>
      <ul>
          <li>
            <a target="_blank" href="https://blog.csdn.net/assiduous_me">
              <img src="https://profile-avatar.csdnimg.cn/6be6e98720414a5a90c37f94e605b42c_assiduous_me.jpg!1"
                alt="assiduous_me" class="avatar">
            </a>
            <div class="right-box">
              <div class="new-info-box clearfix">
                <div class="comment-top">
                  <div class="user-box">
                    <a class="name-href" target="_blank"  href="https://blog.csdn.net/assiduous_me">
                      <span class="name ">Don&#39;t_Touch_Me</span></a>
                    <span class="date" title="2021-11-18 22:27:03">2021.11.18</span>
                    <div class="new-comment">大佬，我想请教一个问题，dma-buf heaps框架，driver层创建了一个heap，用户层可以去通过接口在heap中分配buffer，得到fd，然后通过ioctl传递给其他的驱动去共享这个buffer，这个是对的吧？也就是一个heap可以申请多个buffer？
然后我还想问一下，ion和这个dma-buf heaps的区别是什么呢？这个dma-buf heaps 和ion框架底层都是用的dma-buf框架，在上面进行的封装的？</div>
                  </div>
                </div>
              </div>
            </div>
          </li>
          <li >
            <ul>
                <li>
                  <a target="_blank" href="https://blog.csdn.net/MeloSydneyUni">
                    <img src="https://profile-avatar.csdnimg.cn/default.jpg!1"
                      alt="MeloSydneyUni" class="avatar">
                  </a>
                  <div class="right-box">
                    <div class="new-info-box clearfix">
                      <div class="comment-top">
                        <div class="user-box">
                          <a class="name-href" target="_blank"  href="https://blog.csdn.net/MeloSydneyUni">
                            <span class="name ">Melo__</span><span class="text">回复</span><span class="nick-name">Don&#39;t_Touch_Me</span>
                          </a>
                          <span class="date" title="2023-11-02 15:56:03">2023.11.02</span>
                          <div class="new-comment">1. 可以 同一个进程内 fd共享的, 所以一定确保同一个进程
2. ion只有一个文件, heaps有多个文件, 对于不同的文件可以通过selinux控制访问权限, 更安全, 而且 heaps是upstream的方案 更加稳定</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </li>
            </ul>
          </li>
      </ul>
      <ul>
          <li>
            <a target="_blank" href="https://blog.csdn.net/assiduous_me">
              <img src="https://profile-avatar.csdnimg.cn/6be6e98720414a5a90c37f94e605b42c_assiduous_me.jpg!1"
                alt="assiduous_me" class="avatar">
            </a>
            <div class="right-box">
              <div class="new-info-box clearfix">
                <div class="comment-top">
                  <div class="user-box">
                    <a class="name-href" target="_blank"  href="https://blog.csdn.net/assiduous_me">
                      <span class="name ">Don&#39;t_Touch_Me</span></a>
                    <span class="date" title="2021-11-17 11:38:45">2021.11.17</span>
                    <div class="new-comment">这里缺失用户态代码，没有用户态代码获得 exporter 的fd，从而传递给 importer 的 ioctl，这些用例是没有办法执行的</div>
                  </div>
                </div>
              </div>
            </div>
          </li>
          <li >
            <ul>
                <li>
                  <a target="_blank" href="https://blog.csdn.net/assiduous_me">
                    <img src="https://profile-avatar.csdnimg.cn/6be6e98720414a5a90c37f94e605b42c_assiduous_me.jpg!1"
                      alt="assiduous_me" class="avatar">
                  </a>
                  <div class="right-box">
                    <div class="new-info-box clearfix">
                      <div class="comment-top">
                        <div class="user-box">
                          <a class="name-href" target="_blank"  href="https://blog.csdn.net/assiduous_me">
                            <span class="name ">Don&#39;t_Touch_Me</span><span class="text">回复</span><span class="nick-name">何小龙</span>
                          </a>
                          <span class="date" title="2021-11-18 22:22:20">2021.11.18</span>
                          <div class="new-comment">有道理[face]emoji:009.png[/face]，我刚才看了一下这个，发现前面有这个用户态代码</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </li>
                <li>
                  <a target="_blank" href="https://blog.csdn.net/hexiaolong2009">
                    <img src="https://profile-avatar.csdnimg.cn/13e0f6961a114b5cb09ff65ed930a667_hexiaolong2009.jpg!1"
                      alt="hexiaolong2009" class="avatar">
                  </a>
                  <div class="right-box">
                    <div class="new-info-box clearfix">
                      <div class="comment-top">
                        <div class="user-box">
                          <a class="name-href" target="_blank"  href="https://blog.csdn.net/hexiaolong2009">
                            <span class="name ">何小龙</span><span class="text">回复</span><span class="nick-name">Don&#39;t_Touch_Me</span>
                          </a>
                          <span class="date" title="2021-11-18 20:17:08">2021.11.18</span>
                          <div class="new-comment">如果你看过前面几篇文章，就不会留下这个评论了。</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </li>
            </ul>
          </li>
      </ul>
    </div>
              <div class="recommend-box insert-baidu-box recommend-box-style ">
                <div class="recommend-item-box no-index" style="display:none"></div>
<div class="recommend-item-box type_blog clearfix" data-url="https://devpress.csdn.net/v1/article/detail/153252082"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.2","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~YuanLiJiHua~activity-2-153252082-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"2","strategy":"2~default~YuanLiJiHua~activity","dest":"https://devpress.csdn.net/v1/article/detail/153252082"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://devpress.csdn.net/v1/article/detail/153252082" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.2","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~YuanLiJiHua~activity-2-153252082-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"2","strategy":"2~default~YuanLiJiHua~activity","dest":"https://devpress.csdn.net/v1/article/detail/153252082"}'  data-report-query='spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7Eactivity-2-153252082-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7Eactivity-2-153252082-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1"><em>Linux</em> <em>Page</em> Table（页表）</div>
					<div class="tag">最新发布</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/hezuijiudexiaobai" target="_blank"><span class="blog-title">喝醉酒的小白</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">10-14</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					609
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://devpress.csdn.net/v1/article/detail/153252082" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.2","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~YuanLiJiHua~activity-2-153252082-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"2","strategy":"2~default~YuanLiJiHua~activity","dest":"https://devpress.csdn.net/v1/article/detail/153252082"}'  data-report-query='spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7Eactivity-2-153252082-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7Eactivity-2-153252082-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">数据库<em>Page</em>Table 影响建议措施性能收益Oracle大SGA导致页表庞大、TLB miss频繁启用Huge<em>Page</em>s提升5~20%MySQL<em>Buf</em>fer Pool大量页映射、TLB flush频繁禁用THP+启用Large <em>Page</em>s提升5~15%openGauss多进程共享内存 + 页表复制使用Huge<em>Page</em>s、优化NUMA提升10~15%✅<em>Page</em> Table 对数据库性能的影响在于 &ldquo;页映射层的复杂度&rdquo; 和 &ldquo;TLB效率&rdquo;，</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/weixin_39999532/article/details/116925217"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.3","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-3-116925217-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"3","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/weixin_39999532/article/details/116925217"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/weixin_39999532/article/details/116925217" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.3","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-3-116925217-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"3","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/weixin_39999532/article/details/116925217"}'  data-report-query='spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-3-116925217-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-3-116925217-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1"><em>linux</em>中分配<em>dma</em>缓冲区,<em>linux</em>3.10.65 <em>DMA</em>缓冲区分配失败</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/weixin_39999532" target="_blank"><span class="blog-title">weixin_39999532的博客</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">05-14</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					749
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/weixin_39999532/article/details/116925217" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.3","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-3-116925217-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"3","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/weixin_39999532/article/details/116925217"}'  data-report-query='spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-3-116925217-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-3-116925217-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">今天调试tp固件升级的时候，<em>DMA</em>缓冲区按照以前android4.2<em>(</em><em>内核</em><em>版本</em>不记得了<em>)</em>,<em>DMA</em>缓冲区的申请方式，发现老是申请失败。原来的申请方式如下：static u8 *I2C<em>DMA</em><em>Buf</em>_va = NULL;<em>dma</em>_addr_t I2C<em>DMA</em><em>Buf</em>_pa =NULL;I2C<em>DMA</em><em>Buf</em>_va = <em>(</em>u8 *<em>)</em><em>dma</em>_<em>alloc</em>_coherent<em>(</em>NULL, FTS_<em>DMA</em>_<em>BUF</em>_SIZE,...</div>
			</a>
		</div>
	</div>
</div>
		<dl id="recommend-item-box-tow" class="recommend-item-box type_blog clearfix">
			
		</dl>
<div class="recommend-item-box type_blog clearfix" data-url="https://devpress.csdn.net/v1/article/detail/102596744"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.4","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-4-102596744-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"4","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596744"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://devpress.csdn.net/v1/article/detail/102596744" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.4","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-4-102596744-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"4","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596744"}'  data-report-query='spm=1001.2101.3001.6650.4&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-4-102596744-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-4-102596744-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1"><em>dma</em><em>-</em><em>buf</em> <em>由浅入深</em>（一） &mdash;&mdash; 最简单的 <em>dma</em><em>-</em><em>buf</em> 驱动程序</div>
					<div class="tag">热门推荐</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/hexiaolong2009" target="_blank"><span class="blog-title">hexiaolong2009的专栏</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">11-26</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					8万+
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://devpress.csdn.net/v1/article/detail/102596744" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.4","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-4-102596744-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"4","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596744"}'  data-report-query='spm=1001.2101.3001.6650.4&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-4-102596744-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-4-102596744-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">如果你和我一样，是一位从事Android多媒体底层开发的工程师，那么你对 <em>dma</em><em>-</em><em>buf</em> 这个词语一定不会陌生，因为不管是 Video、Camera 还是 Display、GPU，它们的<em>buf</em>fer都来自于ION，而 ION 正是基于 <em>dma</em><em>-</em><em>buf</em> 实现的。

假如你对 <em>dma</em><em>-</em><em>buf</em> 的理解并不深刻，又期望找个时间来彻底公关一下，那么很高兴，这几篇文章一定能让你对 <em>dma</em><em>-</em><em>buf</em> 有个更深入、更透彻的理解。</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://devpress.csdn.net/v1/article/detail/102596772"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.5","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-5-102596772-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"5","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596772"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://devpress.csdn.net/v1/article/detail/102596772" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.5","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-5-102596772-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"5","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596772"}'  data-report-query='spm=1001.2101.3001.6650.5&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-5-102596772-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-5-102596772-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1"><em>dma</em><em>-</em><em>buf</em> <em>由浅入深</em>（三） &mdash;&mdash; map attachment</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/hexiaolong2009" target="_blank"><span class="blog-title">hexiaolong2009的专栏</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">11-26</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					3万+
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://devpress.csdn.net/v1/article/detail/102596772" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.5","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-5-102596772-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"5","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596772"}'  data-report-query='spm=1001.2101.3001.6650.5&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-5-102596772-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-5-102596772-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">在上一篇《kmap/vmap》中，我们学习了如何使用 <em>CPU</em> 在 kernel 空间访问 <em>dma</em><em>-</em><em>buf</em> 物理内存，但如果使用<em>CPU</em>直接去访问 memory，那么性能会大大降低。因此，<em>dma</em><em>-</em><em>buf</em> 在<em>内核</em>中出现频率最高的还是它的 <em>dma</em>_<em>buf</em>_attach<em>(</em><em>)</em> 和 <em>dma</em>_<em>buf</em>_map_attachment<em>(</em><em>)</em> 接口。本篇我们就一起来学习如何通过这两个 API 来实现 <em>DMA</em> 硬件对 <em>dma</em><em>-</em><em>buf</em> 物理内存的访问。</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/relax33/article/details/128319124"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.6","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~OPENSEARCH~Rate-6-128319124-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"6","strategy":"2~default~OPENSEARCH~Rate","dest":"https://blog.csdn.net/relax33/article/details/128319124"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/relax33/article/details/128319124" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.6","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~OPENSEARCH~Rate-6-128319124-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"6","strategy":"2~default~OPENSEARCH~Rate","dest":"https://blog.csdn.net/relax33/article/details/128319124"}'  data-report-query='spm=1001.2101.3001.6650.6&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7ERate-6-128319124-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7ERate-6-128319124-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">ION to SMMU</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/relax33" target="_blank"><span class="blog-title">relax33的博客</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">12-14</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					1484
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/relax33/article/details/128319124" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.6","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~OPENSEARCH~Rate-6-128319124-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"6","strategy":"2~default~OPENSEARCH~Rate","dest":"https://blog.csdn.net/relax33/article/details/128319124"}'  data-report-query='spm=1001.2101.3001.6650.6&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7ERate-6-128319124-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7ERate-6-128319124-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">ION <em>DMA</em><em>-</em><em>BUF</em> IOMMU</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/zhangxinjieli3/article/details/125026537"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.7","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~OPENSEARCH~Rate-7-125026537-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"7","strategy":"2~default~OPENSEARCH~Rate","dest":"https://blog.csdn.net/zhangxinjieli3/article/details/125026537"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/zhangxinjieli3/article/details/125026537" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.7","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~OPENSEARCH~Rate-7-125026537-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"7","strategy":"2~default~OPENSEARCH~Rate","dest":"https://blog.csdn.net/zhangxinjieli3/article/details/125026537"}'  data-report-query='spm=1001.2101.3001.6650.7&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7ERate-7-125026537-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7ERate-7-125026537-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1"><em>Linux</em>学习总结</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/zhangxinjieli3" target="_blank"><span class="blog-title">zhangxinjieli3的专栏</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">05-29</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					1183
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/zhangxinjieli3/article/details/125026537" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.7","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~OPENSEARCH~Rate-7-125026537-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"7","strategy":"2~default~OPENSEARCH~Rate","dest":"https://blog.csdn.net/zhangxinjieli3/article/details/125026537"}'  data-report-query='spm=1001.2101.3001.6650.7&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7ERate-7-125026537-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7ERate-7-125026537-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">虽然工作中一直在写用户层代码，但工作中也凭兴趣看下kernel代码，提高自己，毕竟懂得越多，对工作和业务了解、架构设计就越有帮助，而且很多东西看到最后都是借助kernel实现，所以掌握kernel也是非常必要。从业以来也陆陆续续看过，解决一些疑问，但没有记录，也没想的特别明白，还有一些是没有认真想过的东西，现在有时间认真思考下，本文相当于十万个为什么，记录我对<em>linux</em><em>内核</em>和驱动的一些学习。

<em>内核</em>概述

<em>linux</em><em>内核</em>分为进程管理系统 、内存管理系统 、 i/o管理系统 和文件管理系统四.........</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_wk_aigc_column clearfix" data-url="https://wenku.csdn.net/column/6k4dk1iqew"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.8","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Position-18-2qr6q87w5b-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"18","strategy":"2~default~OPENSEARCH~Position","dest":"https://wenku.csdn.net/column/6k4dk1iqew"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://wenku.csdn.net/column/6k4dk1iqew" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.8","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Position-18-2qr6q87w5b-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"18","strategy":"2~default~OPENSEARCH~Position","dest":"https://wenku.csdn.net/column/6k4dk1iqew"}'  data-report-query='spm=1001.2101.3001.6650.8&utm_medium=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-18-2qr6q87w5b-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-18-2qr6q87w5b-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1"><em>DMA</em>引擎集成与性能调优：提升ARM平台数据吞吐的3大关键技术</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info display-flex">
					<span class="info-block"></span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://wenku.csdn.net/column/6k4dk1iqew" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.8","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Position-18-2qr6q87w5b-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"18","strategy":"2~default~OPENSEARCH~Position","dest":"https://wenku.csdn.net/column/6k4dk1iqew"}'  data-report-query='spm=1001.2101.3001.6650.8&utm_medium=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-18-2qr6q87w5b-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-18-2qr6q87w5b-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">在现代嵌入式系统中，<em>DMA</em>（Direct Memory <em>Access</em>）引擎已成为高效数据传输的核心组件。它通过绕开<em>CPU</em>直接在外设与内存间搬运数据，显著降低处理器负载，提升系统整体吞吐能力。尤其在ARM架构平台中，<em>DMA</em>与AMBA总线、...</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_wk_aigc_column clearfix" data-url="https://wenku.csdn.net/column/7zogw0t61g"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.9","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Position-18-2qr6q87w5b-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"18","strategy":"2~default~OPENSEARCH~Position","dest":"https://wenku.csdn.net/column/7zogw0t61g"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://wenku.csdn.net/column/7zogw0t61g" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.9","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Position-18-2qr6q87w5b-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"18","strategy":"2~default~OPENSEARCH~Position","dest":"https://wenku.csdn.net/column/7zogw0t61g"}'  data-report-query='spm=1001.2101.3001.6650.9&utm_medium=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-18-2qr6q87w5b-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-18-2qr6q87w5b-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1"><em>DMA</em>传输总失败？深度解读地址对齐与缓存一致性这2大调试盲区</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info display-flex">
					<span class="info-block"></span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://wenku.csdn.net/column/7zogw0t61g" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.9","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Position-18-2qr6q87w5b-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"18","strategy":"2~default~OPENSEARCH~Position","dest":"https://wenku.csdn.net/column/7zogw0t61g"}'  data-report-query='spm=1001.2101.3001.6650.9&utm_medium=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-18-2qr6q87w5b-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-18-2qr6q87w5b-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">[<em>DMA</em>传输总失败？深度解读地址对齐与缓存一致性这2大调试盲区]<em>(</em>https://res.cloudinary.com/witspry/image/upload/witscad/public/content/courses/computer<em>-</em>architecture/<em>dma</em>c<em>-</em>functional<em>-</em>components.png<em>)</em> # 1. <em>DMA</em>...</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_wk_aigc_column clearfix" data-url="https://wenku.csdn.net/column/68ajp00upz"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.10","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Position-18-2qr6q87w5b-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"18","strategy":"2~default~OPENSEARCH~Position","dest":"https://wenku.csdn.net/column/68ajp00upz"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://wenku.csdn.net/column/68ajp00upz" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.10","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Position-18-2qr6q87w5b-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"18","strategy":"2~default~OPENSEARCH~Position","dest":"https://wenku.csdn.net/column/68ajp00upz"}'  data-report-query='spm=1001.2101.3001.6650.10&utm_medium=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-18-2qr6q87w5b-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-18-2qr6q87w5b-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1"><em>DMA</em>与硬件加速协同优化：提升嵌入式图像处理效率的3大瓶颈突破方案</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info display-flex">
					<span class="info-block"></span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://wenku.csdn.net/column/68ajp00upz" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.10","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Position-18-2qr6q87w5b-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"18","strategy":"2~default~OPENSEARCH~Position","dest":"https://wenku.csdn.net/column/68ajp00upz"}'  data-report-query='spm=1001.2101.3001.6650.10&utm_medium=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-18-2qr6q87w5b-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-18-2qr6q87w5b-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">在高性能嵌入式系统中，<em>DMA</em>（Direct Memory <em>Access</em>）与硬件加速器的协同工作已成为提升数据吞吐、降低<em>CPU</em>负载的关键路径。传统中断驱动的数据传输模式已难以满足图像、视频等高带宽应用场景的实时</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://devpress.csdn.net/v1/article/detail/102596761"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.11","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-11-102596761-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"11","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596761"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://devpress.csdn.net/v1/article/detail/102596761" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.11","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-11-102596761-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"11","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596761"}'  data-report-query='spm=1001.2101.3001.6650.11&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-11-102596761-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-11-102596761-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1"><em>dma</em><em>-</em><em>buf</em> <em>由浅入深</em>（二） &mdash;&mdash; kmap / vmap</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/hexiaolong2009" target="_blank"><span class="blog-title">hexiaolong2009的专栏</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">11-26</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					3万+
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://devpress.csdn.net/v1/article/detail/102596761" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.11","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-11-102596761-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"11","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596761"}'  data-report-query='spm=1001.2101.3001.6650.11&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-11-102596761-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-11-102596761-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">在上一篇《最简单的 <em>dma</em><em>-</em><em>buf</em> 驱动程序》中，我们学习了编写 <em>dma</em><em>-</em><em>buf</em> 驱动程序的三个基本步骤，即 <em>dma</em>_<em>buf</em>_ops 、 <em>dma</em>_<em>buf</em>_export_info、 <em>dma</em>_<em>buf</em>_export<em>(</em><em>)</em>。在本篇中，我们将在 exporter<em>-</em>dummy 驱动的基础上，对其 <em>dma</em>_<em>buf</em>_ops 的 kmap / vmap 接口进行扩展，以此来演示这两个接口的使用方法。</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://devpress.csdn.net/v1/article/detail/102596791"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.12","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-12-102596791-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"12","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596791"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://devpress.csdn.net/v1/article/detail/102596791" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.12","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-12-102596791-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"12","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596791"}'  data-report-query='spm=1001.2101.3001.6650.12&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-12-102596791-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-12-102596791-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1"><em>dma</em><em>-</em><em>buf</em> <em>由浅入深</em>（四） &mdash;&mdash; mmap</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/hexiaolong2009" target="_blank"><span class="blog-title">hexiaolong2009的专栏</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">11-26</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					2万+
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://devpress.csdn.net/v1/article/detail/102596791" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.12","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-12-102596791-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"12","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596791"}'  data-report-query='spm=1001.2101.3001.6650.12&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-12-102596791-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-12-102596791-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">前面的两篇文章《<em>dma</em><em>-</em><em>buf</em> <em>由浅入深</em>（二） &mdash;&mdash; kmap/vmap》和《<em>dma</em><em>-</em><em>buf</em> <em>由浅入深</em>（三） &mdash;&mdash; map attachment》都是在 kernel space 对 <em>dma</em><em>-</em><em>buf</em> 进行访问的，本篇我们将一起来学习，如何在 user space 访问 <em>dma</em><em>-</em><em>buf</em>。当然，user space 访问 <em>dma</em><em>-</em><em>buf</em> 也属于 <em>CPU</em> <em>Access</em> 的一种。</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/weixin_42262944/article/details/119853846"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.13","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-13-119853846-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"13","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/weixin_42262944/article/details/119853846"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/weixin_42262944/article/details/119853846" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.13","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-13-119853846-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"13","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/weixin_42262944/article/details/119853846"}'  data-report-query='spm=1001.2101.3001.6650.13&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-13-119853846-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-13-119853846-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1"><em>Linux</em>内存子系统&mdash;&mdash;分配物理页面（<em>alloc</em>_<em>page</em>s）</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/weixin_42262944" target="_blank"><span class="blog-title">深海</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">08-22</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					6760
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/weixin_42262944/article/details/119853846" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.13","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-13-119853846-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"13","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/weixin_42262944/article/details/119853846"}'  data-report-query='spm=1001.2101.3001.6650.13&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-13-119853846-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-13-119853846-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1"><em>Linux</em>内存子系统&mdash;&mdash;分配物理页面（<em>alloc</em>_<em>page</em>s）分配页面
&emsp;&emsp;<em>内核</em>中常用的分配物理内存页面的接口函数是<em>alloc</em>_<em>page</em>s<em>(</em><em>)</em>，用于分配一个或多个连续的物理页面，分配的页面个数只能是2的整数次幂。
&emsp;&emsp;诸如vm<em>alloc</em>、get_user_<em>page</em>s、以及缺页中断中分配页面，都是通过该接口分配的物理页面。
分配页面
&emsp;&emsp;<em>alloc</em>_<em>page</em>s函数有两个参数，一个是分配掩码gfp_mask，另一个是分配阶数order。
[include/<em>linux</em>/gfp.h]
#define <em>alloc</em>_p</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://kernel.blog.csdn.net/article/details/52704844"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.14","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-14-52704844-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"14","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://kernel.blog.csdn.net/article/details/52704844"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://kernel.blog.csdn.net/article/details/52704844" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.14","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-14-52704844-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"14","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://kernel.blog.csdn.net/article/details/52704844"}'  data-report-query='spm=1001.2101.3001.6650.14&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-14-52704844-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-14-52704844-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1"><em>alloc</em>_<em>page</em>分配内存空间<em>-</em><em>-</em><em>Linux</em>内存管理<em>(</em>十<em>七</em><em>)</em></div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/gatieme" target="_blank"><span class="blog-title">OSKernelLAB(gatieme)</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">09-29</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					2万+
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://kernel.blog.csdn.net/article/details/52704844" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.14","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-14-52704844-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"14","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://kernel.blog.csdn.net/article/details/52704844"}'  data-report-query='spm=1001.2101.3001.6650.14&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-14-52704844-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-14-52704844-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">日期
  <em>内核</em><em>版本</em>
  架构
  作者
  GitHub
  CSDN  2016<em>-</em>09<em>-</em>29
  <em>Linux</em><em>-</em>4.7
  X86 &amp; arm
  gatieme
  <em>Linux</em>DeviceDrivers
  <em>Linux</em>内存管理
1  前景回顾在<em>内核</em>初始化完成之后, 内存管理的责任就由伙伴系统来承担. 伙伴系统基于一种相对简单然而令人吃惊的强大算法.<em>Linux</em><em>内核</em>使用二进制伙伴算法来管理和分配物理内</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/wxc20062006/article/details/44458611"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.15","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-15-44458611-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"15","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/wxc20062006/article/details/44458611"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/wxc20062006/article/details/44458611" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.15","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-15-44458611-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"15","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/wxc20062006/article/details/44458611"}'  data-report-query='spm=1001.2101.3001.6650.15&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-15-44458611-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-15-44458611-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1"><em>alloc</em>_<em>page</em>函数分析</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/wxc20062006" target="_blank"><span class="blog-title">wxc20062006的专栏</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">03-19</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					1594
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/wxc20062006/article/details/44458611" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.15","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-15-44458611-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"15","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/wxc20062006/article/details/44458611"}'  data-report-query='spm=1001.2101.3001.6650.15&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-15-44458611-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-15-44458611-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">http://blog.chinaunix.net/uid<em>-</em>20729583<em>-</em>id<em>-</em>1884604.html

/*
&nbsp;*下面的<em>alloc</em>_<em>page</em>s<em>(</em>gfp_mask,order<em>)</em>函数用来请求2^order个连续的页框
&nbsp;*/&nbsp;
#define <em>alloc</em>_<em>page</em>s<em>(</em>gfp_mask, order<em>)</em> \
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<em>alloc</em>_<em>page</em>s_node<em>(</em>numa_</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/m0_74282605/article/details/128686931"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.16","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-16-128686931-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"16","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/m0_74282605/article/details/128686931"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/m0_74282605/article/details/128686931" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.16","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-16-128686931-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"16","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/m0_74282605/article/details/128686931"}'  data-report-query='spm=1001.2101.3001.6650.16&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-16-128686931-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-16-128686931-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1"><em>alloc</em>_<em>page</em>分配内存</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/m0_74282605" target="_blank"><span class="blog-title">m0_74282605的博客</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">01-14</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					773
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/m0_74282605/article/details/128686931" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.16","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-16-128686931-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"16","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/m0_74282605/article/details/128686931"}'  data-report-query='spm=1001.2101.3001.6650.16&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-16-128686931-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-16-128686931-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">_get_free_<em>page</em>s调用<em>alloc</em>_<em>page</em>s完成内存分配, 而<em>alloc</em>_<em>page</em>s又借助于<em>alloc</em>_<em>page</em>s_node，__get_free_<em>page</em>s函数的定义在mm/<em>page</em>_<em>alloc</em>.c。<em>alloc</em>_flags和gfp_mask之间的区别，gfp_mask是使用<em>alloc</em>_<em>page</em>s申请内存时所传递的申请标记，而<em>alloc</em>_flags是在内存管理子系统内部使用的另一个标记。函数直接将自己的所有信息传递给__<em>alloc</em>_<em>page</em>s_nodemask来完成内存的分配。</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/z20230508/article/details/139264705"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.17","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-17-139264705-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"17","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/z20230508/article/details/139264705"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/z20230508/article/details/139264705" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.17","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-17-139264705-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"17","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/z20230508/article/details/139264705"}'  data-report-query='spm=1001.2101.3001.6650.17&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-17-139264705-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-17-139264705-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1"><em>linux</em> 之<em>dma</em>_<em>buf</em> <em>(</em>7<em>)</em><em>-</em> <em>alloc</em> <em>page</em> <em>版本</em></div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/z20230508" target="_blank"><span class="blog-title">z20230508的博客</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">05-28</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					596
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/z20230508/article/details/139264705" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.17","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-17-139264705-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"17","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/z20230508/article/details/139264705"}'  data-report-query='spm=1001.2101.3001.6650.17&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-17-139264705-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-17-139264705-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">exporter 所分配的内存都是通过 kz<em>alloc</em><em>(</em><em>)</em> 来分配的。本篇我们换个方式，使用 <em>alloc</em>_<em>page</em><em>(</em><em>)</em> 来分配内存。</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_chatgpt clearfix" data-url="https://wenku.csdn.net/answer/2qr6q87w5b"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.18","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Position-18-2qr6q87w5b-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"18","strategy":"2~default~OPENSEARCH~Position","dest":"https://wenku.csdn.net/answer/2qr6q87w5b"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://wenku.csdn.net/answer/2qr6q87w5b" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.18","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Position-18-2qr6q87w5b-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"18","strategy":"2~default~OPENSEARCH~Position","dest":"https://wenku.csdn.net/answer/2qr6q87w5b"}'  data-report-query='spm=1001.2101.3001.6650.18&utm_medium=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-18-2qr6q87w5b-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-18-2qr6q87w5b-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1"><em>dma</em><em>-</em><em>buf</em> <em>由浅入深</em></div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info display-flex">
					<span class="info-block">05-13</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://wenku.csdn.net/answer/2qr6q87w5b" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.18","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Position-18-2qr6q87w5b-blog-102596845.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382499823_87507\"}","dist_request_id":"1766382499823_87507","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"18","strategy":"2~default~OPENSEARCH~Position","dest":"https://wenku.csdn.net/answer/2qr6q87w5b"}'  data-report-query='spm=1001.2101.3001.6650.18&utm_medium=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-18-2qr6q87w5b-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-18-2qr6q87w5b-blog-102596845.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">### 什么是 <em>DMA</em><em>-</em><em>BUF</em>？  <em>DMA</em><em>-</em><em>BUF</em> 是 <em>Linux</em> <em>内核</em>中的一个框架，用于跨子系统的缓冲区共享。它允许不同的硬件子系统（如 GPU、DSP 或其他外设）之间高效地共享内存数据，从而减少不必要的复制操作并提高性能[^5]。  <em>-</em><em>-</em><em>-</em>...</div>
			</a>
		</div>
	</div>
</div>
              </div>
<div class="blog-footer-bottom" style="margin-top:10px;"></div>
<script src="https://g.csdnimg.cn/common/csdn-footer/csdn-footer.js" data-isfootertrack="false" type="text/javascript"></script>
<script type="text/javascript">
    window.csdn.csdnFooter.options = {
        el: '.blog-footer-bottom',
        type: 2
    }
</script>          </main>
<aside class="blog_container_aside ">
<div id="asideProfile" class="aside-box active">
    <div class="profile-intro d-flex">
        <div class="avatar-box d-flex justify-content-center flex-column">
            <a href="https://blog.csdn.net/hexiaolong2009" target="_blank" data-report-click='{"mod":"popu_379","spm":"3001.4121","dest":"https://blog.csdn.net/hexiaolong2009","ab":"new"}'>
                <img src="https://profile-avatar.csdnimg.cn/13e0f6961a114b5cb09ff65ed930a667_hexiaolong2009.jpg!1" class="avatar_pic">
            </a>
        </div>
        <div class="user-info d-flex flex-column profile-intro-name-box">
            <div class="profile-intro-name-boxTop">
                <a href="https://blog.csdn.net/hexiaolong2009" target="_blank" class="" id="uid" title="何小龙" data-report-click='{"mod":"popu_379","spm":"3001.4122","dest":"https://blog.csdn.net/hexiaolong2009","ab":"new"}'>
                    <span class="name" username="hexiaolong2009">何小龙</span>
                </a>
            </div>
            <div class="profile-intro-name-boxFooter-new">
              <p class="profile-intro-name-leve">
                <span>
                  博客等级
                </span>
                  <img class="level" src="https://csdnimg.cn/identity/blog6.png">
              </p>
                <span class="profile-intro-name-years" title="已加入 CSDN 16年">码龄16年</span>
               
            </div>
        </div>
    </div>
    <div class="profile-intro-Identity-information">
        <p class="profile-information-box">
          <img class="information-img" data-report-click='{"spm":"3001.4296"}' src="https://i-operation.csdnimg.cn/images/586260c6ecd54b20be60ced2d94df1d8.png" alt="">
          <span>领域专家: 嵌入式与硬件开发技术领域</span>
        </p>

    </div>
    <div class="profile-intro-rank-information">
      <dl>
        <a href="https://blog.csdn.net/hexiaolong2009" data-report-click='{"mod":"1598321000_001","spm":"3001.4310"}' data-report-query="t=1">  
            <dd><span >84</span></dd>
            <dt>原创</dt>
        </a>
      </dl>
       <dl title="1941">
        <dd>1941</dd>
        <dt>点赞</dt>
      </dl>
       <dl title="4922">
        <dd>4922</dd>
        <dt>收藏</dt>
      </dl>
      <dl id="fanBox" title="3270">
        <dd><span id="fan">3270</span></dd>
        <dt>粉丝</dt>
      </dl>
    </div>
    <div class="profile-intro-name-boxOpration">
        <div class="opt-letter-watch-box"> 
            <a class="personal-watch bt-button" id="btnAttent" >关注</a>  
        </div>
        <div class="opt-letter-watch-box">
        <a rel="nofollow" class="bt-button personal-letter" href="https://im.csdn.net/chat/hexiaolong2009" target="_blank" rel="noopener">私信</a>
        </div>
    </div>
</div>



<div id="asideHotArticle" class="aside-box">
	<h3 class="aside-title">热门文章</h3>
	<div class="aside-content">
		<ul class="hotArticle-list">
			<li>
				<a href="https://blog.csdn.net/hexiaolong2009/article/details/83720940" target="_blank"  data-report-click='{"mod":"popu_541","spm":"3001.4139","dest":"https://blog.csdn.net/hexiaolong2009/article/details/83720940","ab":"new"}'>
				DRM（Direct Rendering Manager）学习简介
					<img src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					<span class="read">121270</span>
                </a>
			</li>
			<li>
				<a href="https://blog.csdn.net/hexiaolong2009/article/details/102596744" target="_blank"  data-report-click='{"mod":"popu_541","spm":"3001.4139","dest":"https://blog.csdn.net/hexiaolong2009/article/details/102596744","ab":"new"}'>
				dma-buf 由浅入深（一） —— 最简单的 dma-buf 驱动程序
					<img src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					<span class="read">83296</span>
                </a>
			</li>
			<li>
				<a href="https://blog.csdn.net/hexiaolong2009/article/details/83721242" target="_blank"  data-report-click='{"mod":"popu_541","spm":"3001.4139","dest":"https://blog.csdn.net/hexiaolong2009/article/details/83721242","ab":"new"}'>
				最简单的DRM应用程序 （single-buffer）
					<img src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					<span class="read">64173</span>
                </a>
			</li>
			<li>
				<a href="https://blog.csdn.net/hexiaolong2009/article/details/89810355" target="_blank"  data-report-click='{"mod":"popu_541","spm":"3001.4139","dest":"https://blog.csdn.net/hexiaolong2009/article/details/89810355","ab":"new"}'>
				DRM 驱动程序开发（开篇）
					<img src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					<span class="read">48022</span>
                </a>
			</li>
			<li>
				<a href="https://blog.csdn.net/hexiaolong2009/article/details/79319512" target="_blank"  data-report-click='{"mod":"popu_541","spm":"3001.4139","dest":"https://blog.csdn.net/hexiaolong2009/article/details/79319512","ab":"new"}'>
				LCD显示异常分析——撕裂(tear effect)
					<img src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					<span class="read">42349</span>
                </a>
			</li>
		</ul>
	</div>
</div>
<div id="asideCategory" class="aside-box aside-box-column ">
    <h3 class="aside-title">分类专栏</h3>
    <div class="aside-content" id="aside-content">
        <ul>
            <li>
                <a class="clearfix special-column-name"  href="https://blog.csdn.net/hexiaolong2009/category_9281458.html" data-report-click='{"mod":"popu_537","spm":"3001.4137","strategy":"pc付费专栏左侧入口","dest":"https://blog.csdn.net/hexiaolong2009/category_9281458.html","ab":"new"}'>
                    <div class="special-column-bar "></div>
                    <img src="https://i-blog.csdnimg.cn/blog_column_migrate/c03d60f8b24740d58f94a787024155be.png?x-oss-process=image/resize,m_fixed,h_64,w_64" alt="" onerror="this.src='https://i-blog.csdnimg.cn/columns/default/20201014180756922.png?x-oss-process=image/resize,m_fixed,h_64,w_64'">
                    <span class="title oneline">
                        DRM (Direct Rendering Manager)
                    </span>
                </a>
                <span class="special-column-num">29篇</span>
            </li>
            <li>
                <a class="clearfix special-column-name"  href="https://blog.csdn.net/hexiaolong2009/category_10331964.html" data-report-click='{"mod":"popu_537","spm":"3001.4137","strategy":"pc付费专栏左侧入口","dest":"https://blog.csdn.net/hexiaolong2009/category_10331964.html","ab":"new"}'>
                    <div class="special-column-bar "></div>
                    <img src="https://i-blog.csdnimg.cn/columns/default/20201014180756757.png?x-oss-process=image/resize,m_fixed,h_64,w_64" alt="" onerror="this.src='https://i-blog.csdnimg.cn/columns/default/20201014180756922.png?x-oss-process=image/resize,m_fixed,h_64,w_64'">
                    <span class="title oneline">
                        Linux Graphics 周刊
                    </span>
                </a>
                <span class="special-column-num">10篇</span>
            </li>
            <li>
                <a class="clearfix special-column-name"  href="https://blog.csdn.net/hexiaolong2009/category_9813335.html" data-report-click='{"mod":"popu_537","spm":"3001.4137","strategy":"pc付费专栏左侧入口","dest":"https://blog.csdn.net/hexiaolong2009/category_9813335.html","ab":"new"}'>
                    <div class="special-column-bar "></div>
                    <img src="https://i-blog.csdnimg.cn/columns/default/20201014180756927.png?x-oss-process=image/resize,m_fixed,h_64,w_64" alt="" onerror="this.src='https://i-blog.csdnimg.cn/columns/default/20201014180756922.png?x-oss-process=image/resize,m_fixed,h_64,w_64'">
                    <span class="title oneline">
                        Wayland
                    </span>
                </a>
            </li>
            <li>
                <a class="clearfix special-column-name"  href="https://blog.csdn.net/hexiaolong2009/category_9705063.html" data-report-click='{"mod":"popu_537","spm":"3001.4137","strategy":"pc付费专栏左侧入口","dest":"https://blog.csdn.net/hexiaolong2009/category_9705063.html","ab":"new"}'>
                    <div class="special-column-bar "></div>
                    <img src="https://i-blog.csdnimg.cn/columns/default/20201014180756927.png?x-oss-process=image/resize,m_fixed,h_64,w_64" alt="" onerror="this.src='https://i-blog.csdnimg.cn/columns/default/20201014180756922.png?x-oss-process=image/resize,m_fixed,h_64,w_64'">
                    <span class="title oneline">
                        GPU
                    </span>
                </a>
                <span class="special-column-num">6篇</span>
            </li>
            <li>
                <a class="clearfix special-column-name"  href="https://blog.csdn.net/hexiaolong2009/category_7583191.html" data-report-click='{"mod":"popu_537","spm":"3001.4137","strategy":"pc付费专栏左侧入口","dest":"https://blog.csdn.net/hexiaolong2009/category_7583191.html","ab":"new"}'>
                    <div class="special-column-bar "></div>
                    <img src="https://i-blog.csdnimg.cn/columns/default/20201014180756927.png?x-oss-process=image/resize,m_fixed,h_64,w_64" alt="" onerror="this.src='https://i-blog.csdnimg.cn/columns/default/20201014180756922.png?x-oss-process=image/resize,m_fixed,h_64,w_64'">
                    <span class="title oneline">
                        Android
                    </span>
                </a>
                <span class="special-column-num">4篇</span>
            </li>
        </ul>
    </div>
</div>
  <div class="article-previous" id="article-previous">
      <dl data-report-click='{"spm":"3001.10752","extend1":"上一篇"}' data-report-view='{"spm":"3001.10752","extend1":"上一篇"}'>
          <dt>
              上一篇：
          </dt>
          <dd>
            <a href="https://blog.csdn.net/hexiaolong2009/article/details/102596825" data-report-query="spm=3001.10752">
              dma-buf 由浅入深（六） —— begin / end cpu_access
            </a>
          </dd>
      </dl>
      <dl class="next" data-report-click='{"spm":"3001.10796","extend1":"下一篇"}' data-report-view='{"spm":"3001.10796","extend1":"下一篇"}'>
          <dt>
            下一篇：
          </dt>
          <dd >
            <a href="https://blog.csdn.net/hexiaolong2009/article/details/103795381" data-report-query="spm=3001.10796">
              dma-buf 由浅入深（八） —— ION 简化版
            </a>
          </dd>
      </dl>
  </div>
<div id="asideHotArticle" class="aside-box">
	<h3 class="aside-title">大家在看</h3>
	<div class="aside-content">
		<ul class="hotArticle-list">
			<li>
				<a href="https://blog.csdn.net/2509_93766561/article/details/156148540" target="_blank"  data-report-click='{"spm":"3001.10093","dest":"https://blog.csdn.net/2509_93766561/article/details/156148540","strategy":"202_1052723-3681419_RCMD","ab":"new"}' data-report-view='{"spm":"3001.10093","dest":"https://blog.csdn.net/2509_93766561/article/details/156148540","strategy":"202_1052723-3681419_RCMD","ab":"new"}'>
				斗南花市点燃年轻旅游潮
        </a>
			</li>
			<li>
				<a href="https://blog.csdn.net/qq_39980997/article/details/156115532" target="_blank"  data-report-click='{"spm":"3001.10093","dest":"https://blog.csdn.net/qq_39980997/article/details/156115532","strategy":"202_1052723-3681433_RCMD","ab":"new"}' data-report-view='{"spm":"3001.10093","dest":"https://blog.csdn.net/qq_39980997/article/details/156115532","strategy":"202_1052723-3681433_RCMD","ab":"new"}'>
				初中生键盘指法入门指南
					<img src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					<span class="read">210</span>
        </a>
			</li>
			<li>
				<a href="https://blog.csdn.net/qkh1234567/article/details/156146565" target="_blank"  data-report-click='{"spm":"3001.10093","dest":"https://blog.csdn.net/qkh1234567/article/details/156146565","strategy":"202_1052723-3681410_RCMD","ab":"new"}' data-report-view='{"spm":"3001.10093","dest":"https://blog.csdn.net/qkh1234567/article/details/156146565","strategy":"202_1052723-3681410_RCMD","ab":"new"}'>
				大模型面试高频题解析
        </a>
			</li>
			<li>
				<a href="https://blog.csdn.net/weixin_47121252/article/details/156143506" target="_blank"  data-report-click='{"spm":"3001.10093","dest":"https://blog.csdn.net/weixin_47121252/article/details/156143506","strategy":"202_1052723-3681438_RCMD","ab":"new"}' data-report-view='{"spm":"3001.10093","dest":"https://blog.csdn.net/weixin_47121252/article/details/156143506","strategy":"202_1052723-3681438_RCMD","ab":"new"}'>
				Python实现斐波那契数列
					<img src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					<span class="read">41</span>
        </a>
			</li>
			<li>
				<a href="https://blog.csdn.net/2503_94684233/article/details/156148931" target="_blank"  data-report-click='{"spm":"3001.10093","dest":"https://blog.csdn.net/2503_94684233/article/details/156148931","strategy":"202_1052723-3681429_RCMD","ab":"new"}' data-report-view='{"spm":"3001.10093","dest":"https://blog.csdn.net/2503_94684233/article/details/156148931","strategy":"202_1052723-3681429_RCMD","ab":"new"}'>
				校园圈子系统源码：多端创业利器
        </a>
			</li>
		</ul>
	</div>
</div>
<div id="asideArchive" class="aside-box" style="display:block!important; width:300px;">
    <h3 class="aside-title">最新文章</h3>
    <div class="aside-content">
        <ul class="inf_list clearfix">
            <li class="clearfix">
            <a href="https://blog.csdn.net/hexiaolong2009/article/details/111876034" target="_blank" data-report-click='{"mod":"popu_382","spm":"3001.4136","dest":"https://blog.csdn.net/hexiaolong2009/article/details/111876034","ab":"left"}' data-report-view='{"mod":"popu_382","spm":"3001.4136","dest":"https://blog.csdn.net/hexiaolong2009/article/details/111876034","ab":"left"}'>LWN 翻译：Atomic Mode Setting 设计简介（下）</a>
            </li>
            <li class="clearfix">
            <a href="https://blog.csdn.net/hexiaolong2009/article/details/112498800" target="_blank" data-report-click='{"mod":"popu_382","spm":"3001.4136","dest":"https://blog.csdn.net/hexiaolong2009/article/details/112498800","ab":"left"}' data-report-view='{"mod":"popu_382","spm":"3001.4136","dest":"https://blog.csdn.net/hexiaolong2009/article/details/112498800","ab":"left"}'>Linux Graphics 周刊（第 9 期）</a>
            </li>
            <li class="clearfix">
            <a href="https://blog.csdn.net/hexiaolong2009/article/details/107661938" target="_blank" data-report-click='{"mod":"popu_382","spm":"3001.4136","dest":"https://blog.csdn.net/hexiaolong2009/article/details/107661938","ab":"left"}' data-report-view='{"mod":"popu_382","spm":"3001.4136","dest":"https://blog.csdn.net/hexiaolong2009/article/details/107661938","ab":"left"}'>LWN 翻译：Atomic Mode Setting 设计简介（上）</a>
            </li>
        </ul>
        <div class="archive-bar"></div>
        <div class="archive-box">
                <div class="archive-list-item"><a href="https://blog.csdn.net/hexiaolong2009?type=blog&amp;year=2021&amp;month=02" target="_blank" data-report-click='{"mod":"popu_538","spm":"3001.4138","ab":"new","dest":"https://blog.csdn.net/hexiaolong2009?type=blog&amp;year=2021&amp;month=02"}'><span class="year">2021年</span><span class="num">2篇</span></a></div>
                <div class="archive-list-item"><a href="https://blog.csdn.net/hexiaolong2009?type=blog&amp;year=2020&amp;month=12" target="_blank" data-report-click='{"mod":"popu_538","spm":"3001.4138","ab":"new","dest":"https://blog.csdn.net/hexiaolong2009?type=blog&amp;year=2020&amp;month=12"}'><span class="year">2020年</span><span class="num">28篇</span></a></div>
                <div class="archive-list-item"><a href="https://blog.csdn.net/hexiaolong2009?type=blog&amp;year=2019&amp;month=11" target="_blank" data-report-click='{"mod":"popu_538","spm":"3001.4138","ab":"new","dest":"https://blog.csdn.net/hexiaolong2009?type=blog&amp;year=2019&amp;month=11"}'><span class="year">2019年</span><span class="num">15篇</span></a></div>
                <div class="archive-list-item"><a href="https://blog.csdn.net/hexiaolong2009?type=blog&amp;year=2018&amp;month=12" target="_blank" data-report-click='{"mod":"popu_538","spm":"3001.4138","ab":"new","dest":"https://blog.csdn.net/hexiaolong2009?type=blog&amp;year=2018&amp;month=12"}'><span class="year">2018年</span><span class="num">7篇</span></a></div>
                <div class="archive-list-item"><a href="https://blog.csdn.net/hexiaolong2009?type=blog&amp;year=2015&amp;month=04" target="_blank" data-report-click='{"mod":"popu_538","spm":"3001.4138","ab":"new","dest":"https://blog.csdn.net/hexiaolong2009?type=blog&amp;year=2015&amp;month=04"}'><span class="year">2015年</span><span class="num">12篇</span></a></div>
                <div class="archive-list-item"><a href="https://blog.csdn.net/hexiaolong2009?type=blog&amp;year=2014&amp;month=03" target="_blank" data-report-click='{"mod":"popu_538","spm":"3001.4138","ab":"new","dest":"https://blog.csdn.net/hexiaolong2009?type=blog&amp;year=2014&amp;month=03"}'><span class="year">2014年</span><span class="num">5篇</span></a></div>
                <div class="archive-list-item"><a href="https://blog.csdn.net/hexiaolong2009?type=blog&amp;year=2013&amp;month=12" target="_blank" data-report-click='{"mod":"popu_538","spm":"3001.4138","ab":"new","dest":"https://blog.csdn.net/hexiaolong2009?type=blog&amp;year=2013&amp;month=12"}'><span class="year">2013年</span><span class="num">28篇</span></a></div>
        </div>
    </div>
</div>
    <!-- 详情页显示目录 -->
<!--文章目录-->
<div id="asidedirectory" class="aside-box">
    <div class='groupfile groupfile-active' id="directory">
        <h3 class="aside-title">目录</h3>
        <div class="align-items-stretch group_item" id="align-items-stretch">
            <div class="pos-box">
            <div class="scroll-box">
                <div class="toc-box"></div>
            </div>
            </div>
        </div>
          <p class="flexible-btn-new active" id="flexible-btn-groupfile" data-report-click='{"spm":"3001.10780","strategy":"展开全部"}' data-minheight="117px" data-maxheight="446px" data-fbox="#align-items-stretch"><span class="text">展开全部</span> <img class="look-more" src="https://csdnimg.cn/release/blogv2/dist/pc/img/arrowup-line-bot-White.png" alt=""></p>
          <p class="flexible-btn-new-close active" data-report-click='{"spm":"3001.10780","strategy":"收起"}'  data-minheight="117px" data-maxheight="446px" data-fbox="#align-items-stretch"><span class="text">收起</span> <img class="look-more" src="https://csdnimg.cn/release/blogv2/dist/pc/img/arrowup-line-top-White.png" alt=""></p>
    </div>
</div>
<div class="gitcode-qc-left-box aside-box" data-report-click='{"spm":"3001.11256", "extra":"{\"position\":\"left\"}"}'></div>
</aside>
<script>
	$("a.flexible-btn").click(function(){
		$(this).parents('div.aside-box').removeClass('flexible-box');
		$(this).parents("p.text-center").remove();
	})
</script>
<script type="text/javascript"  src="https://g.csdnimg.cn/user-tooltip/2.7/user-tooltip.js"></script>
<script type="text/javascript"  src="https://g.csdnimg.cn/user-medal/2.0.0/user-medal.js"></script>        </div>
<div class="recommend-right align-items-stretch clearfix" id="rightAside" data-type="recommend">
    <aside class="recommend-right_aside">
            <div class="rightside-fixed-hide">
      </div>
        <div id="recommend-right" >
          <div class='flex-column aside-box groupfile groupfile-active ' id="groupfile">
              <div class="groupfile-div">
              <h3 class="aside-title">目录</h3>
              <div class="align-items-stretch group_item" id="align-items-stretch-right">
                  <div class="pos-box">
                      <div class="scroll-box">
                          <div class="toc-box"></div>
                      </div>
                  </div>
              </div>
                <p class="flexible-btn-new" id="flexible-btn-groupfile" data-report-click='{"spm":"3001.10782","strategy":"展开全部"}' data-traigger="true" data-minheight="117px" data-maxheight="446px" data-fbox="#align-items-stretch-right"><span class="text">展开全部</span> <img class="look-more" src="https://csdnimg.cn/release/blogv2/dist/pc/img/arrowup-line-bot-White.png" alt=""></p>
                <p class="flexible-btn-new-close close" data-report-click='{"spm":"3001.10782","strategy":"收起"}' data-traigger="true"  data-minheight="117px" data-maxheight="446px" data-fbox="#align-items-stretch-right"><span class="text">收起</span> <img class="look-more" src="https://csdnimg.cn/release/blogv2/dist/pc/img/arrowup-line-top-White.png" alt=""></p>
              </div>
          </div>
          <div class="gitcode-qc-right-box aside-box" data-report-click='{"spm":"3001.11256", "extra":"{\"position\":\"right\"}"}'></div>
  <div class="article-previous" id="article">
      <dl data-report-click='{"spm":"3001.10752","extend1":"上一篇"}' data-report-view='{"spm":"3001.10752","extend1":"上一篇"}'>
          <dt>
              上一篇：
          </dt>
          <dd>
            <a href="https://blog.csdn.net/hexiaolong2009/article/details/102596825" data-report-query="spm=3001.10752">
              dma-buf 由浅入深（六） —— begin / end cpu_access
            </a>
          </dd>
      </dl>
      <dl class="next" data-report-click='{"spm":"3001.10796","extend1":"下一篇"}' data-report-view='{"spm":"3001.10796","extend1":"下一篇"}'>
          <dt>
            下一篇：
          </dt>
          <dd >
            <a href="https://blog.csdn.net/hexiaolong2009/article/details/103795381" data-report-query="spm=3001.10796">
              dma-buf 由浅入深（八） —— ION 简化版
            </a>
          </dd>
      </dl>
  </div>
          <div class='aside-box kind_person d-flex flex-column flexible-box-new' >
                  <h3 class="aside-title">分类专栏</h3>
                  <div class="align-items-stretch kindof_item" id="kind_person_column">
                      <div class="aside-content" id="aside-content-column">
                          <ul>
                              <li>
                                  <a class="clearfix special-column-name"  href="https://blog.csdn.net/hexiaolong2009/category_9281458.html" data-report-click='{"mod":"popu_537","spm":"1001.2101.3001.4137","strategy":"pc付费专栏左侧入口","dest":"https://blog.csdn.net/hexiaolong2009/category_9281458.html","ab":"new"}'>
                                      <div class="special-column-bar "></div>
                                      <img src="https://i-blog.csdnimg.cn/blog_column_migrate/c03d60f8b24740d58f94a787024155be.png?x-oss-process=image/resize,m_fixed,h_64,w_64" alt="" onerror="this.src='https://i-blog.csdnimg.cn/columns/default/20201014180756922.png?x-oss-process=image/resize,m_fixed,h_64,w_64'">
                                      <span class="">
                                          DRM (Direct Rendering Manager)
                                      </span>
                                  </a>
                                  <span class="special-column-num">29篇</span>
                              </li>
                              <li>
                                  <a class="clearfix special-column-name"  href="https://blog.csdn.net/hexiaolong2009/category_10331964.html" data-report-click='{"mod":"popu_537","spm":"1001.2101.3001.4137","strategy":"pc付费专栏左侧入口","dest":"https://blog.csdn.net/hexiaolong2009/category_10331964.html","ab":"new"}'>
                                      <div class="special-column-bar "></div>
                                      <img src="https://i-blog.csdnimg.cn/columns/default/20201014180756757.png?x-oss-process=image/resize,m_fixed,h_64,w_64" alt="" onerror="this.src='https://i-blog.csdnimg.cn/columns/default/20201014180756922.png?x-oss-process=image/resize,m_fixed,h_64,w_64'">
                                      <span class="">
                                          Linux Graphics 周刊
                                      </span>
                                  </a>
                                  <span class="special-column-num">10篇</span>
                              </li>
                              <li>
                                  <a class="clearfix special-column-name"  href="https://blog.csdn.net/hexiaolong2009/category_9813335.html" data-report-click='{"mod":"popu_537","spm":"1001.2101.3001.4137","strategy":"pc付费专栏左侧入口","dest":"https://blog.csdn.net/hexiaolong2009/category_9813335.html","ab":"new"}'>
                                      <div class="special-column-bar "></div>
                                      <img src="https://i-blog.csdnimg.cn/columns/default/20201014180756927.png?x-oss-process=image/resize,m_fixed,h_64,w_64" alt="" onerror="this.src='https://i-blog.csdnimg.cn/columns/default/20201014180756922.png?x-oss-process=image/resize,m_fixed,h_64,w_64'">
                                      <span class="">
                                          Wayland
                                      </span>
                                  </a>
                              </li>
                              <li>
                                  <a class="clearfix special-column-name"  href="https://blog.csdn.net/hexiaolong2009/category_9705063.html" data-report-click='{"mod":"popu_537","spm":"1001.2101.3001.4137","strategy":"pc付费专栏左侧入口","dest":"https://blog.csdn.net/hexiaolong2009/category_9705063.html","ab":"new"}'>
                                      <div class="special-column-bar "></div>
                                      <img src="https://i-blog.csdnimg.cn/columns/default/20201014180756927.png?x-oss-process=image/resize,m_fixed,h_64,w_64" alt="" onerror="this.src='https://i-blog.csdnimg.cn/columns/default/20201014180756922.png?x-oss-process=image/resize,m_fixed,h_64,w_64'">
                                      <span class="">
                                          GPU
                                      </span>
                                  </a>
                                  <span class="special-column-num">6篇</span>
                              </li>
                              <li>
                                  <a class="clearfix special-column-name"  href="https://blog.csdn.net/hexiaolong2009/category_7583191.html" data-report-click='{"mod":"popu_537","spm":"1001.2101.3001.4137","strategy":"pc付费专栏左侧入口","dest":"https://blog.csdn.net/hexiaolong2009/category_7583191.html","ab":"new"}'>
                                      <div class="special-column-bar "></div>
                                      <img src="https://i-blog.csdnimg.cn/columns/default/20201014180756927.png?x-oss-process=image/resize,m_fixed,h_64,w_64" alt="" onerror="this.src='https://i-blog.csdnimg.cn/columns/default/20201014180756922.png?x-oss-process=image/resize,m_fixed,h_64,w_64'">
                                      <span class="">
                                          Android
                                      </span>
                                  </a>
                                  <span class="special-column-num">4篇</span>
                              </li>
                          </ul>
                      </div>
                        <p class="text-center">
                          <a class="flexible-btn-new" data-report-click='{"spm":"3001.10783","strategy":"展开全部"}' data-traigger="true" data-maxheight="0" data-minheight="208px" data-fbox="#aside-content-column" data-flag="flag"><span class="text">展开全部</span> <img class="look-more" src="https://csdnimg.cn/release/blogv2/dist/pc/img/arrowup-line-bot-White.png" alt=""></a>
                          <a class="flexible-btn-new-close" data-report-click='{"spm":"3001.10783","strategy":"收起"}'data-traigger="true"  data-minheight="208px" data-fbox="#aside-content-column" data-scroll="true" data-flag="flag"><span class="text">收起</span> <img class="look-more" src="https://csdnimg.cn/release/blogv2/dist/pc/img/arrowup-line-top-White.png" alt=""></a>
                        </p>
                  </div>
          </div>
        </div>
    </aside>
</div>

<div class="recommend-right1  align-items-stretch clearfix" id="rightAsideConcision" data-type="recommend">
    <aside class="recommend-right_aside">
        <div id="recommend-right-concision" >
            <div class='flex-column aside-box groupfile' id="groupfileConcision">
                <div class="groupfile-div1">
                <h3 class="aside-title">目录</h3>
                <div class="align-items-stretch group_item">
                    <div class="pos-box">
                        <div class="scroll-box">
                            <div class="toc-box"></div>
                        </div>
                    </div>
                </div>
                </div>
            </div>
        </div>
    </aside>
</div>

      </div>
      <div class="mask-dark"></div>
        <script type="text/javascript">
        var timert = setInterval(function() {
          sideToolbar = $(".csdn-side-toolbar");
          if (sideToolbar.length > 0) {
            sideToolbar.css('cssText', 'bottom:64px !important;')
            clearInterval(timert);
          }
        }, 200);
        </script>
      <div class="skin-boxshadow"></div>
      <div class="directory-boxshadow"></div>
<div class="comment-side-box-shadow comment-side-tit-close" id="commentSideBoxshadow">
<div class="comment-side-content">
	<div class="comment-side-tit">
		<div class="comment-side-tit-count">评论&nbsp;<span class="count">6</span></div>
	<img class="comment-side-tit-close" src="https://csdnimg.cn/release/blogv2/dist/pc/img/closeBt.png"></div>
  <div id="pcCommentSideBox" class="comment-box comment-box-new2 }" style="display:'block'}">
	
    <div class="comment-edit-box d-flex">
      <div class="user-img">
      </div>
      <form id="commentform">
        <textarea class="comment-content" name="comment_content" id="comment_content" placeholder="欢迎高质量的评论，低质的评论会被折叠" maxlength="1000"></textarea>
        <div class="comment-reward-box" style="background-image: url('https://img-home.csdnimg.cn/images/20230131025301.png');">
          <a class="btn-remove-reward"></a>
          <div class="form-reward-box">
            <div class="info">
              成就一亿技术人!
            </div>
            <div class="price-info">
              拼手气红包<span class="price">6.0元</span>
            </div>
          </div>
        </div>
        <div class="comment-operate-box">
          <div class="comment-operate-l">
            <span id="tip_comment" class="tip">还能输入<em>1000</em>个字符</span>
          </div>
          <div class="comment-operate-c">
            &nbsp;
          </div>
          <div class="comment-operate-r">
            <div class="comment-operate-item comment-reward">
              <img class="comment-operate-img" data-url="https://csdnimg.cn/release/blogv2/dist/pc/img/" src="https://csdnimg.cn/release/blogv2/dist/pc/img/commentReward.png" alt="红包">
              <span class="comment-operate-tip">添加红包</span>
            </div>
            <div class="comment-operate-item comment-emoticon">
              <img class="comment-operate-img" data-url="https://csdnimg.cn/release/blogv2/dist/pc/img/" src="https://csdnimg.cn/release/blogv2/dist/pc/img/commentEmotionIcon.png" alt="表情包">
              <span class="comment-operate-tip">插入表情</span>
              <div class="comment-emoticon-box comment-operate-isshow">
                <div class="comment-emoticon-img-box"></div>
              </div>
            </div>
            <div class="comment-operate-item comment-code">
              <img class="comment-operate-img" data-url="https://csdnimg.cn/release/blogv2/dist/pc/img/" src="https://csdnimg.cn/release/blogv2/dist/pc/img/commentCodeIcon.png" alt="表情包">
              <span class="comment-operate-tip">代码片</span>
              <div class="comment-code-box comment-operate-isshow">
                <ul id="commentCode">
                  <li><a data-code="html">HTML/XML</a></li>
                  <li><a data-code="objc">objective-c</a></li>
                  <li><a data-code="ruby">Ruby</a></li>
                  <li><a data-code="php">PHP</a></li>
                  <li><a data-code="csharp">C</a></li>
                  <li><a data-code="cpp">C++</a></li>
                  <li><a data-code="javascript">JavaScript</a></li>
                  <li><a data-code="python">Python</a></li>
                  <li><a data-code="java">Java</a></li>
                  <li><a data-code="css">CSS</a></li>
                  <li><a data-code="sql">SQL</a></li>
                  <li><a data-code="plain">其它</a></li>
                </ul>
              </div>
            </div>
            <div class="comment-operate-item">
              <input type="hidden" id="comment_replyId" name="comment_replyId">
              <input type="hidden" id="article_id" name="article_id" value="102596845">
              <input type="hidden" id="comment_userId" name="comment_userId" value="">
              <input type="hidden" id="commentId" name="commentId" value="">
              <a data-report-click='{"mod":"1582594662_003","spm":"1001.2101.3001.4227","ab":"new"}'>
              <input type="submit" class="btn-comment btn-comment-input" value="评论">
              </a>
            </div>
          </div>
        </div>
      </form>
    </div>
		<div class="comment-list-container">
			<div class="comment-list-box comment-operate-item">
			</div>
			<div id="lookGoodComment" class="look-good-comment side-look-comment">
				<a class="look-more-comment">查看更多评论<img src="https://csdnimg.cn/release/blogv2/dist/pc/img/commentArrowDownWhite.png" alt=""></a>
			</div>
			<div id="lookFlodComment" class="look-flod-comment">
					<span class="count"></span>&nbsp;条评论被折叠&nbsp;<a class="look-more-flodcomment">查看</a>
			</div>
			<div class="opt-box text-center">
				<div class="btn btn-sm btn-link-blue" id="btnMoreComment"></div>
			</div>
		</div>
	</div>
	<div id="pcFlodCommentSideBox" class="pc-flodcomment-sidebox">
		<div class="comment-fold-tit"><span id="lookUnFlodComment" class="back"><img src="https://csdnimg.cn/release/blogv2/dist/pc/img/commentArrowLeftWhite.png" alt=""></span>被折叠的&nbsp;<span class="count"></span>&nbsp;条评论
		 <a href="https://blogdev.blog.csdn.net/article/details/122245662" class="tip" target="_blank">为什么被折叠?</a>
		 <a href="https://bbs.csdn.net/forums/FreeZone" class="park" target="_blank">
		 <img src="https://csdnimg.cn/release/blogv2/dist/pc/img/iconPark.png">到【灌水乐园】发言</a>                                
		</div>
		<div class="comment-fold-content"></div>
		<div id="lookBadComment" class="look-bad-comment side-look-comment">
			<a class="look-more-comment">查看更多评论<img src="https://csdnimg.cn/release/blogv2/dist/pc/img/commentArrowDownWhite.png" alt=""></a>
		</div>
	</div>
</div>
<div class="comment-rewarddialog-box">
  <div class="form-box">
    <div class="title-box">
      添加红包
      <a class="btn-form-close"></a>
    </div>
    <form id="commentRewardForm">
      <div class="ipt-box">
        <label for="txtName">祝福语</label>
        <div class="ipt-btn-box">
          <input type="text" name="name" id="txtName" autocomplete="off" maxlength="50">
          <a class="btn-ipt btn-random"></a>
        </div>
        <p class="notice">请填写红包祝福语或标题</p>
      </div>
      <div class="ipt-box">
        <label for="txtSendAmount">红包数量</label>
        <div class="ipt-txt-box">
          <input type="text" name="sendAmount" maxlength="4" id="txtSendAmount" placeholder="请填写红包数量(最小10个)" autocomplete="off">
          <span class="after-txt">个</span>
        </div>
        <p class="notice">红包个数最小为10个</p>
      </div>
      <div class="ipt-box">
        <label for="txtMoney">红包总金额</label>
        <div class="ipt-txt-box error">
          <input type="text" name="money" maxlength="5" id="txtMoney" placeholder="请填写总金额(最低5元)" autocomplete="off">
          <span class="after-txt">元</span>
        </div>
        <p class="notice">红包金额最低5元</p>
      </div>
      <div class="balance-info-box">
        <label>余额支付</label>
        <div class="balance-info">
          当前余额<span class="balance">3.43</span>元
          <a href="https://i.csdn.net/#/wallet/balance/recharge" class="link-charge" target="_blank">前往充值 ></a>
        </div>
      </div>
      <div class="opt-box">
        <div class="pay-info">
          需支付：<span class="price">10.00</span>元
        </div>
        <button type="button" class="ml-auto btn-cancel">取消</button>
        <button type="button" class="ml8 btn-submit" disabled="true">确定</button>
      </div>
    </form>
  </div>
</div>
<div class="rr-guide-box">
  <div class="rr-first-box">
    <img src="https://csdnimg.cn/release/blogv2/dist/pc/img/guideRedReward02.png" alt="">
    <button class="btn-guide-known next">下一步</button>
  </div>
  <div class="rr-second-box">
    <img src="https://csdnimg.cn/release/blogv2/dist/pc/img/guideRedReward03.png" alt="">
    <button class="btn-guide-known known">知道了</button>
  </div>
</div>
</div>

<div class="redEnvolope" id="redEnvolope">
  <div class="env-box">
    <div class="env-container">
      <div class="pre-open" id="preOpen">
        <div class="top">
          <header>
            <img class="clearTpaErr" :src="redpacketAuthor.avatar" alt="" />
            <div class="author">成就一亿技术人!</div>
          </header>
          <div class="bot-icon"></div>
        </div>
        <footer>
          <div class="red-openbtn open-start"></div>
          <div class="tip">
            领取后你会自动成为博主和红包主的粉丝
            <a class="rule" target="_blank">规则</a>
          </div>
        </footer>
      </div>
      <div class="opened" id="opened">
        <div class="bot-icon">
          <header>
            <a class="creatorUrl" href="" target="_blank">
              <img class="clearTpaErr" src="https://profile-avatar.csdnimg.cn/default.jpg!2" alt="" />
            </a>
            <div class="author">
              <div class="tt">hope_wisdom</div> 发出的红包
            </div>
          </header>
        </div>
        <div class="receive-box">
          <header></header>
          <div class="receive-list">
          </div>
        </div>
      </div>
    </div>
    <div class="close-btn"></div>
  </div>
</div>
<div id="rewardNew" class="reward-popupbox-new">
	<p class="rewad-title">打赏作者<span class="reward-close"><img src="https://csdnimg.cn/release/blogv2/dist/pc/img/closeBt.png"></span></p>
	<dl class="profile-box">
		<dd>
		<a href="https://blog.csdn.net/hexiaolong2009" data-report-click='{"mod":"popu_379","dest":"https://blog.csdn.net/hexiaolong2009","ab":"new"}'>
			<img src="https://profile-avatar.csdnimg.cn/13e0f6961a114b5cb09ff65ed930a667_hexiaolong2009.jpg!1" class="avatar_pic">
		</a>
		</dd>
		<dt>
			<p class="blog-name">何小龙</p>
			<p class="blog-discript">你的鼓励将是我创作的最大动力</p>
		</dt>
	</dl>
	<div class="reward-box-new">
			<div class="reward-content"><div class="reward-right"></div></div>
	</div>
	<div class="money-box">
    <span class="choose-money choosed" data-id="1">¥1</span>
    <span class="choose-money " data-id="2">¥2</span>
    <span class="choose-money " data-id="4">¥4</span>
    <span class="choose-money " data-id="6">¥6</span>
    <span class="choose-money " data-id="10">¥10</span>
    <span class="choose-money " data-id="20">¥20</span>
	</div>
	<div class="sure-box">
		<div class="sure-box-money">
			<div class="code-box">
				<div class="code-num-box">
					<span class="code-name">扫码支付：</span><span class="code-num">¥1</span>
				</div>
				<div class="code-img-box">
					<div class="renovate">
					<img src="https://csdnimg.cn/release/blogv2/dist/pc/img/pay-time-out.png">
					<span>获取中</span>
					</div>
				</div>
				<div class="code-pay-box">
					<img src="https://csdnimg.cn/release/blogv2/dist/pc/img/newWeiXin.png" alt="">
					<img src="https://csdnimg.cn/release/blogv2/dist/pc/img/newZhiFuBao.png" alt="">
					<span>扫码支付</span>
				</div>
			</div>
		</div>
		<div class="sure-box-blance">
			<p class="tip">您的余额不足，请更换扫码支付或<a target="_blank" data-report-click='{"mod":"1597646289_003","spm":"1001.2101.3001.4302"}' href="https://i.csdn.net/#/wallet/balance/recharge?utm_source=RewardVip" class="go-invest">充值</a></p>
			<p class="is-have-money"><a class="reward-sure">打赏作者</a></p>
		</div>
	</div>
</div>
      
      <div class="pay-code">
      <div class="pay-money">实付<span class="pay-money-span" data-nowprice='' data-oldprice=''>元</span></div>
      <div class="content-blance"><a class="blance-bt" href="javascript:;">使用余额支付</a></div>
      <div class="content-code">
        <div id="payCode" data-id="">
          <div class="renovate">
            <img src="https://csdnimg.cn/release/blogv2/dist/pc/img/pay-time-out.png">
            <span>点击重新获取</span>
          </div>
        </div>
        <div class="pay-style"><span><img src="https://csdnimg.cn/release/blogv2/dist/pc/img/weixin.png"></span><span><img src="https://csdnimg.cn/release/blogv2/dist/pc/img/zhifubao.png"></span><span><img src="https://csdnimg.cn/release/blogv2/dist/pc/img/jingdong.png"></span><span class="text">扫码支付</span></div>
      </div>
      <div class="bt-close">
        <svg t="1567152543821" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="10924" xmlns:xlink="http://www.w3.org/1999/xlink" width="12" height="12">
          <defs>
            <style type="text/css"></style>
          </defs>
          <path d="M512 438.378667L806.506667 143.893333a52.032 52.032 0 1 1 73.6 73.621334L585.621333 512l294.485334 294.485333a52.074667 52.074667 0 0 1-73.6 73.642667L512 585.621333 217.514667 880.128a52.053333 52.053333 0 1 1-73.621334-73.642667L438.378667 512 143.893333 217.514667a52.053333 52.053333 0 1 1 73.621334-73.621334L512 438.378667z" fill="" p-id="10925"></path>
        </svg>
      </div>
      <div class="pay-balance">
        <input type="radio" class="pay-code-radio" data-type="details">
        <span class="span">钱包余额</span>
          <span class="balance" style="color:#FC5531;font-size:14px;">0</span>
          <div class="pay-code-tile">
            <img src="https://csdnimg.cn/release/blogv2/dist/pc/img/pay-help.png" alt="">
            <div class="pay-code-content">
              <div class="span">
                <p class="title">抵扣说明：</p>
                <p> 1.余额是钱包充值的虚拟货币，按照1:1的比例进行支付金额的抵扣。<br> 2.余额无法直接购买下载，可以购买VIP、付费专栏及课程。</p>
              </div>
            </div>
          </div>
      </div>
      <a class="pay-balance-con" href="https://i.csdn.net/#/wallet/balance/recharge" target="_blank"><img src="https://csdnimg.cn/release/blogv2/dist/pc/img/recharge.png" alt=""><span>余额充值</span></a>
    </div>
    <div style="display:none;">
      <img src="" onerror='setTimeout(function(){if(!/(csdn.net|iteye.com|baiducontent.com|googleusercontent.com|360webcache.com|sogoucdn.com|bingj.com|baidu.com)$/.test(window.location.hostname)){window.location.href="\x68\x74\x74\x70\x73\x3a\x2f\x2f\x77\x77\x77\x2e\x63\x73\x64\x6e\x2e\x6e\x65\x74"}},3000);'>
    </div>
    <div class="keyword-dec-box" id="keywordDecBox"></div>
  </body>
  <script src="https://csdnimg.cn/release/blogv2/dist/components/js/axios-83fa28cedf.min.js" type="text/javascript"></script>
  <script src="https://csdnimg.cn/release/blogv2/dist/components/js/pc_wap_highlight-8defd55d6e.min.js" type="text/javascript"></script>
  <script src="https://csdnimg.cn/release/blogv2/dist/components/js/pc_wap_common-3c7b273c43.min.js" type="text/javascript"></script>
  <script src="https://csdnimg.cn/release/blogv2/dist/components/js/edit_copy_code-a22e5c2c2a.min.js" type="text/javascript"></script>
  <script src="https://g.csdnimg.cn/lib/cboxEditor/1.1.6/embed-editor.min.js" type="text/javascript"></script>
  <link rel="stylesheet" href="https://g.csdnimg.cn/lib/cboxEditor/1.1.6/embed-editor.min.css">
  <link rel="stylesheet" href="https://csdnimg.cn/release/blog_editor_html/release1.6.12/ckeditor/plugins/codesnippet/lib/highlight/styles/atom-one-dark.css">
  <script src="https://g.csdnimg.cn/user-accusation/1.0.6/user-accusation.js" type="text/javascript"></script>
  <script>
    // 全局声明
    if (window.csdn === undefined) {
      window.csdn = {};
    }
    var sideToolbarOpt = {}

    $(function() {
      $(document).on('click', "#toolReportBtnHideNormal,#toolReportBtnHide", function() {
        window.csdn.loginBox.key({
          biz: 'blog',
          subBiz: 'other_service',
          cb: function() {
            window.csdn.feedback({
              "type": 'blog',
              "rtype": 'article',
              "rid": articleId,
              "reportedName": username,
              "submitOptions": {
                "title": articleTitle,
                "contentUrl": articleDetailUrl
              },
              "callback": function() {
                showToast({
                  text: "感谢您的举报，我们会尽快审核！",
                  bottom: '10%',
                  zindex: 9000,
                  speed: 500,
                  time: 1500
                })
              }
            })
          }
        })
      });
    })
      window.csdn.sideToolbar = {
        options: {
          ...sideToolbarOpt,
          theme: 'white',
        }
      }
  </script>
    <script src="https://g.csdnimg.cn/baidu-search/1.0.12/baidu-search.js" type="text/javascript"></script>
  <script src="https://csdnimg.cn/release/download/old_static/js/qrcode.js"></script>
  <script src="https://g.csdnimg.cn/lib/qrcode/1.0.0/qrcode.min.js"></script>
  <script src="https://g.csdnimg.cn/user-ordercart/3.0.1/user-ordercart.js" type="text/javascript"></script>
  <script src="https://g.csdnimg.cn/user-ordertip/5.0.3/user-ordertip.js" type="text/javascript"></script>
  <script src="https://g.csdnimg.cn/order-payment/4.0.5/order-payment.js" type="text/javascript"></script>
  <script src="https://csdnimg.cn/release/blogv2/dist/pc/js/common-50b21fafc8.min.js" type="text/javascript"></script>
  <script src="https://csdnimg.cn/release/blogv2/dist/pc/js/detail-ff634bd1bd.min.js" type="text/javascript"></script>
  <script src="https://csdnimg.cn/release/blogv2/dist/pc/js/column-762ba47480.min.js" type="text/javascript"></script>
    <script src="https://g.csdnimg.cn/side-toolbar/3.6/side-toolbar.js" type="text/javascript"></script>
  <script src="https://g.csdnimg.cn/copyright/1.0.4/copyright.js" type="text/javascript"></script>
  <script>
    $(".MathJax").remove();
    if ($('div.markdown_views pre.prettyprint code.hljs').length > 0) {
      $('div.markdown_views')[0].className = 'markdown_views';
    }
  </script>
  <script type="text/javascript" src="https://csdnimg.cn/release/blog_mathjax/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
  <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
      "HTML-CSS": {
        linebreaks: { automatic: true, width: "94%container" },
        imageFont: null
      },
      tex2jax: {
      preview: "none",
      ignoreClass:"title-article"
      },
      mml2jax: {
      preview: 'none'
      }
    });
  </script>
<script type="text/javascript" crossorigin src="https://g.csdnimg.cn/common/csdn-login-box/csdn-login-box.js"></script></html>
