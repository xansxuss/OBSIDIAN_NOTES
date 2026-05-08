    <!DOCTYPE html>
<html lang="zh-CN">

<head>
    <meta charset="utf-8">
    <link rel="canonical" href="https://blog.csdn.net/hexiaolong2009/article/details/102596802"/>
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
    <title>dma-buf 由浅入深（五） —— File_dma-buf fd 传输-CSDN博客</title>
    <script>
      (function(){ 
        var el = document.createElement("script"); 
        el.src = "https://s3a.pstatp.com/toutiao/push.js?1abfa13dfe74d72d41d83c86d240de427e7cac50c51ead53b2e79d40c7952a23ed7716d05b4a0f683a653eab3e214672511de2457e74e99286eb2c33f4428830"; 
        el.id = "ttzz"; 
        var s = document.getElementsByTagName("script")[0]; 
        s.parentNode.insertBefore(el, s);
      })(window)
    </script>
        <meta name="keywords" content="dma-buf fd 传输">
        <meta name="csdn-baidu-search"  content='{"autorun":true,"install":true,"keyword":"dma-buf fd 传输"}'>
    <meta name="description" content="文章浏览阅读2.6w次，点赞22次，收藏63次。在上一篇《dma-buf 由浅入深（四）—— mmap》中，曾提到过 dma_buf_fd() 这个函数，该函数用于创建一个新的 fd，并与 dma-buf 的文件关联起来。本篇我们一起来重点学习 dma-buf 与 file 相关的操作接口，以及它们的注意事项。_dma-buf fd 传输">
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
    <script type="application/ld+json">{"@context":"https://ziyuan.baidu.com/contexts/cambrian.jsonld","@id":"https://blog.csdn.net/hexiaolong2009/article/details/102596802","appid":"1638831770136827","pubDate":"2019-11-26T00:12:33","title":"dma-buf 由浅入深（五） &mdash;&mdash; File_dma-buf fd 传输-CSDN博客","upDate":"2019-11-26T00:12:33"}</script>
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
        var loginUrl = "http://passport.csdn.net/account/login?from=https://blog.csdn.net/hexiaolong2009/article/details/102596802";
        var blogUrl = "https://blog.csdn.net/";
        var starMapUrl = "https://ai.csdn.net";
        var inscodeHost = "https://inscode.csdn.net";
        var paymentBalanceUrl = "https://csdnimg.cn/release/vip-business-components/vipPaymentBalance.js";
        var appBlogDomain = "https://app-blog.csdn.net";
        var avatar = "https://profile-avatar.csdnimg.cn/13e0f6961a114b5cb09ff65ed930a667_hexiaolong2009.jpg!1";
        var isCJBlog = false;
        var isStarMap = false;
        var articleTitle = "dma-buf 由浅入深（五） —— File";
        var articleDesc = "文章浏览阅读2.6w次，点赞22次，收藏63次。在上一篇《dma-buf 由浅入深（四）—— mmap》中，曾提到过 dma_buf_fd() 这个函数，该函数用于创建一个新的 fd，并与 dma-buf 的文件关联起来。本篇我们一起来重点学习 dma-buf 与 file 相关的操作接口，以及它们的注意事项。_dma-buf fd 传输";
        var articleTitles = "dma-buf 由浅入深（五） —— File_dma-buf fd 传输-CSDN博客";
        var nickName = "何小龙";
        var articleDetailUrl = "https://blog.csdn.net/hexiaolong2009/article/details/102596802";
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
            var toolbarSearchExt = '{\"id\":102596802,\"landingWord\":[\"dma-buf fd 传输\"],\"queryWord\":\"\",\"tag\":[\"dma-buf\",\"DRM\",\"内存管理\"],\"title\":\"dma-buf 由浅入深（五） &mdash;&mdash; File\"}';
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
      var articleId = 102596802;
        var privateEduData = [];
        var privateData = ["linux","api","android","初始化","参考资料"];//高亮数组
      var crytojs = "https://csdnimg.cn/release/blogv2/dist/components/js/crytojs-ca5b8bf6ae.min.js";
      var commentscount = 5;
      var commentAuth = 2;
      var curentUrl = "https://blog.csdn.net/hexiaolong2009/article/details/102596802";
      var myUrl = "https://my.csdn.net/";
      var isGitCodeBlog = false;
      var vipActivityIcon = "https://i-operation.csdnimg.cn/images/df6c67fa661c48eba86beaeb64350df0.gif";
      var isOpenSourceBlog = false;
      var isVipArticle = false;
        var highlight = ["file","由浅入深","内存管理","buf","dma","drm","传输","fd","五","(",")","-"];//高亮数组
        var isRecommendModule = true;
          var isBaiduPre = true;
          var baiduCount = 2;
          var setBaiduJsCount = 10;
        var viewCountFormat = 26853;
      var share_card_url = "https://app-blog.csdn.net/share?article_id=102596802&username=hexiaolong2009"
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
      var baiduKey = "dma-buf fd 传输";
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
        var distRequestId = '1766382492343_98303'
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
        var postTime = "2019-11-26 00:12:33"
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
        <h1 class="title-article" id="articleContentId">dma-buf 由浅入深（五） —— File</h1>
      </div>
      <div class="article-info-box">
              <div class="up-time">最新推荐文章于&nbsp;2025-11-01 13:39:13&nbsp;发布</div>
          <div class="article-bar-top">
              <div class="bar-content active">
              <span class="article-type-text original">原创</span>
                    <span class="time blog-postTime" data-time="2019-11-26 00:12:33">最新推荐文章于&nbsp;2025-11-01 13:39:13&nbsp;发布</span>
                <span class="border-dian">·</span>
                <span class="read-count">2.6w 阅读</span>
                <div class="read-count-box is-like like-ab-new" data-type="top">
                  <span class="border-dian">·</span>
                  <img class="article-read-img article-heard-img active" style="display:none" id="is-like-imgactive-new" src="https://csdnimg.cn/release/blogv2/dist/pc/img/newHeart2023Active.png" alt="">
                  <img class="article-read-img article-heard-img" style="display:block" id="is-like-img-new" src="https://csdnimg.cn/release/blogv2/dist/pc/img/newHeart2023Black.png" alt="">
                  <span class="read-count" id="blog-digg-num" style="color:;">
                      22
                  </span>
                </div>
                <span class="border-dian">·</span>
                <a id="blog_detail_zk_collection" class="un-collection" data-report-click='{"mod":"popu_823","spm":"1001.2101.3001.4232","ab":"new"}'>
                  <img class="article-collect-img article-heard-img un-collect-status isdefault" style="display:inline-block" src="https://csdnimg.cn/release/blogv2/dist/pc/img/tobarCollect2.png" alt="">
                  <img class="article-collect-img article-heard-img collect-status isactive" style="display:none" src="https://csdnimg.cn/release/blogv2/dist/pc/img/tobarCollectionActive2.png" alt="">
                  <span class="get-collection">
                      63
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
                          <a rel="nofollow" data-report-query="spm=1001.2101.3001.4223" data-report-click='{"mod":"popu_626","spm":"1001.2101.3001.4223","strategy":"DRM","ab":"new","extra":"{\"searchword\":\"DRM\"}"}' data-report-view='{"mod":"popu_626","spm":"1001.2101.3001.4223","strategy":"DRM","ab":"new","extra":"{\"searchword\":\"DRM\"}"}' class="tag-link-new" href="https://so.csdn.net/so/search/s.do?q=DRM&amp;t=all&amp;o=vip&amp;s=&amp;l=&amp;f=&amp;viparticle=&amp;from_tracking_code=tag_word&amp;from_code=app_blog_art" target="_blank" rel="noopener">#DRM</a>
                          <a rel="nofollow" data-report-query="spm=1001.2101.3001.4223" data-report-click='{"mod":"popu_626","spm":"1001.2101.3001.4223","strategy":"内存管理","ab":"new","extra":"{\"searchword\":\"内存管理\"}"}' data-report-view='{"mod":"popu_626","spm":"1001.2101.3001.4223","strategy":"内存管理","ab":"new","extra":"{\"searchword\":\"内存管理\"}"}' class="tag-link-new" href="https://so.csdn.net/so/search/s.do?q=%E5%86%85%E5%AD%98%E7%AE%A1%E7%90%86&amp;t=all&amp;o=vip&amp;s=&amp;l=&amp;f=&amp;viparticle=&amp;from_tracking_code=tag_word&amp;from_code=app_blog_art" target="_blank" rel="noopener">#内存管理</a>
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
<p>在上一篇<a href="https://blog.csdn.net/hexiaolong2009/article/details/102596791">《dma-buf 由浅入深&#xff08;四&#xff09;—— mmap》</a>中&#xff0c;曾提到过 <code>dma_buf_fd()</code> 这个函数&#xff0c;该函数用于创建一个新的 fd&#xff0c;并与 dma-buf 的文件关联起来。本篇我们一起来重点学习 dma-buf 与 file 相关的操作接口&#xff0c;以及它们的注意事项。</p> 
<h3><a id="file_14"></a>file</h3> 
<p>早在第一篇<a href="https://blog.csdn.net/hexiaolong2009/article/details/102596744">《最简单的 dma-buf 驱动程序》</a>就曾说过&#xff0c;dma-buf 本质上是 buffer 与 file 的结合&#xff0c;不仅如此&#xff0c;<strong>该 file 还是个被 open 过的 file</strong>。从我们调用 <em>dma_buf_export()</em> 开始&#xff0c;这个 file 就已经被 <em>open</em> 了。而且该 file 还是个<strong>匿名文件</strong>&#xff0c;因此应用程序无法通过 <em>fd &#61; open(“name”)</em> 的方式来获取它所对应的 fd&#xff0c;只能依托于 exporter 驱动的 ioctl 接口&#xff0c;通过 <em>dma_buf_fd()</em> 来获取&#xff0c;就像上一篇的<a href="https://blog.csdn.net/hexiaolong2009/article/details/102596791#t3">示例一</a>那样。</p> 
<h3><a id="fd_17"></a>fd</h3> 
<p>如下内核 API 实现了 dma-buf 与 fd 之间的相互转换&#xff1a;</p> 
<ul><li><code>dma_buf_fd()</code>&#xff1a;dma-buf --&gt; new fd</li><li><code>dma_buf_get()</code>&#xff1a;fd --&gt; dma-buf</li></ul> 
<p>通常使用方法如下&#xff1a;</p> 
<pre><code class="prism language-c">fd <span class="token operator">&#61;</span> <span class="token function">dma_buf_fd</span><span class="token punctuation">(</span>dmabuf<span class="token punctuation">)</span><span class="token punctuation">;</span>
dmabuf <span class="token operator">&#61;</span> <span class="token function">dma_buf_get</span><span class="token punctuation">(</span>fd<span class="token punctuation">)</span><span class="token punctuation">;</span>
</code></pre> 
<h3><a id="get__put_29"></a>get / put</h3> 
<p>只要是文件&#xff0c;内部都会有一个引用计数&#xff08;<em>f_count</em>&#xff09;。当使用 <em>dma_buf_export()</em> 函数创建 dma-buf 时&#xff0c;该引用计数被初始化为1&#xff1b;当这个引用计数为0时&#xff0c;则会自动触发 <em>dma_buf_ops</em> 的 <em>release</em> 回调接口&#xff0c;并释放 dma-buf 对象。</p> 
<p>在 linux 内核中操作 file 引用计数的常用函数为 <code>fget()</code> 和 <code>fput()</code>&#xff0c;而 dma-buf 又在此基础上进行了封装&#xff0c;如下&#xff1a;</p> 
<ul><li><code>get_dma_buf()</code></li><li><code>dma_buf_get()</code></li><li><code>dma_buf_put()</code></li></ul> 
<p>为了不让大家混淆&#xff0c;我做了如下表格区分&#xff1a;</p> 
<table><thead><tr><th align="left">函数</th><th align="left">区别</th></tr></thead><tbody><tr><td align="left">get_dma_buf()</td><td align="left">仅引用计数加1</td></tr><tr><td align="left">dma_buf_get()</td><td align="left">引用计数加1&#xff0c;并将 fd 转换成 dma_buf 指针</td></tr><tr><td align="left">dma_buf_put()</td><td align="left">引用计数减1</td></tr><tr><td align="left">dma_buf_fd()</td><td align="left">引用计数不变&#xff0c;仅创建 fd</td></tr></tbody></table>
<h3><a id="release_45"></a>release</h3> 
<p>通常 <em>release</em> 回调接口用来释放 dma-buf 所对应的物理 buffer。当然&#xff0c;凡是所有和该 dma-buf 相关的私有数据也都应该在这里被 free 掉。</p> 
<p>前面说过&#xff0c;只有当 dma-buf 的引用计数递减到0时&#xff0c;才会触发 <em>release</em> 回调接口。因此</p> 
<ul><li><strong>如果不想让你正在使用的 buffer 被突然释放&#xff0c;请提前 get&#xff1b;</strong></li><li><strong>如果想在 kernel space 释放 buffer&#xff0c;请使劲 put&#xff1b;</strong></li><li><strong>如果想从 user space 释放 buffer&#xff0c;请尝试 close&#xff1b;</strong></li></ul> 
<p>这就是为什么在内核设备驱动中&#xff0c;我们会看到那么多 dma-buf get 和 put 的身影。</p> 
<blockquote> 
 <p>这也是为什么在第一篇<a href="https://blog.csdn.net/hexiaolong2009/article/details/102596744">《最简单的 dma-buf 驱动程序》</a>中&#xff0c;一旦 exporter-dummy.ko 被成功加载了&#xff0c;就无法被 rmmod 的原因。因为没有任何程序来修改该 dma-buf 的引用计数&#xff0c;自始自终都保持为1&#xff0c;所以也就无法执行 release 接口&#xff0c;更不会执行 module put。</p> 
</blockquote> 
<h3><a id="_57"></a>示例</h3> 
<p>在前面所有的 exporter 驱动中&#xff0c;都定义了一个 <em>dmabuf_exported</em> 全局变量&#xff0c;方便 importer 驱动通过 extern 关键字来引用。这就造成了 exporter 驱动与 importer 驱动之间的强耦合&#xff0c;不仅编译时 importer 需要依赖 exporter 的文件&#xff0c;就连运行时也要依赖 exporter 模块先加载。</p> 
<p><img src="https://i-blog.csdnimg.cn/blog_migrate/0bf392b27c0a1fa21058997c1fda9f5e.png#pic_center" alt="在这里插入图片描述" width="575" height="397" /></p> 
<p>这次&#xff0c;我们将 <em>dmabuf_exported</em> 全局变量改为 static 静态变量&#xff0c;并借助于 <code>dma_buf_fd()</code> 与 <code>dma_buf_get()</code> 来彻底解除 importer 与 exporter 驱动之间的耦合。</p> 
<h4><a id="exporter__64"></a>exporter 驱动</h4> 
<p>基于上一篇<a href="https://blog.csdn.net/hexiaolong2009/article/details/102596791#t3">示例一</a>中的 exporter 驱动&#xff0c;将 <em>dmabuf_exported</em> 全局变量修改为 static 静态变量&#xff0c;其它代码不做修改。</p> 
<p><a href="https://github.com/hexiaolong2008/sample-code/blob/master/dma-buf/06/exporter-fd.c">exporter-fd.c</a></p> 
<pre><code class="prism language-c"><span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/dma-buf.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/module.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/miscdevice.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/slab.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/uaccess.h&gt;</span></span>

<span class="token keyword">static</span> <span class="token keyword">struct</span> dma_buf <span class="token operator">*</span>dmabuf_exported<span class="token punctuation">;</span>

<span class="token punctuation">.</span><span class="token punctuation">.</span><span class="token punctuation">.</span>

<span class="token keyword">static</span> <span class="token keyword">struct</span> dma_buf <span class="token operator">*</span><span class="token function">exporter_alloc_page</span><span class="token punctuation">(</span><span class="token keyword">void</span><span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token function">DEFINE_DMA_BUF_EXPORT_INFO</span><span class="token punctuation">(</span>exp_info<span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token keyword">struct</span> dma_buf <span class="token operator">*</span>dmabuf<span class="token punctuation">;</span>
	<span class="token keyword">void</span> <span class="token operator">*</span>vaddr<span class="token punctuation">;</span>

	vaddr <span class="token operator">&#61;</span> <span class="token function">kzalloc</span><span class="token punctuation">(</span>PAGE_SIZE<span class="token punctuation">,</span> GFP_KERNEL<span class="token punctuation">)</span><span class="token punctuation">;</span>

	exp_info<span class="token punctuation">.</span>ops <span class="token operator">&#61;</span> <span class="token operator">&amp;</span>exp_dmabuf_ops<span class="token punctuation">;</span>
	exp_info<span class="token punctuation">.</span>size <span class="token operator">&#61;</span> PAGE_SIZE<span class="token punctuation">;</span>
	exp_info<span class="token punctuation">.</span>flags <span class="token operator">&#61;</span> O_CLOEXEC<span class="token punctuation">;</span>
	exp_info<span class="token punctuation">.</span>priv <span class="token operator">&#61;</span> vaddr<span class="token punctuation">;</span>

	dmabuf <span class="token operator">&#61;</span> <span class="token function">dma_buf_export</span><span class="token punctuation">(</span><span class="token operator">&amp;</span>exp_info<span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token function">sprintf</span><span class="token punctuation">(</span>vaddr<span class="token punctuation">,</span> <span class="token string">&#34;hello world!&#34;</span><span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token keyword">return</span> dmabuf<span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token keyword">static</span> <span class="token keyword">long</span> <span class="token function">exporter_ioctl</span><span class="token punctuation">(</span><span class="token keyword">struct</span> file <span class="token operator">*</span>filp<span class="token punctuation">,</span> <span class="token keyword">unsigned</span> <span class="token keyword">int</span> cmd<span class="token punctuation">,</span> <span class="token keyword">unsigned</span> <span class="token keyword">long</span> arg<span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token keyword">int</span> fd <span class="token operator">&#61;</span> <span class="token function">dma_buf_fd</span><span class="token punctuation">(</span>dmabuf_exported<span class="token punctuation">,</span> O_CLOEXEC<span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token function">copy_to_user</span><span class="token punctuation">(</span><span class="token punctuation">(</span><span class="token keyword">int</span> __user <span class="token operator">*</span><span class="token punctuation">)</span>arg<span class="token punctuation">,</span> <span class="token operator">&amp;</span>fd<span class="token punctuation">,</span> <span class="token keyword">sizeof</span><span class="token punctuation">(</span>fd<span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token keyword">return</span> <span class="token number">0</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>
 
<span class="token keyword">static</span> <span class="token keyword">struct</span> file_operations exporter_fops <span class="token operator">&#61;</span> <span class="token punctuation">{<!-- --></span>
	<span class="token punctuation">.</span>owner		<span class="token operator">&#61;</span> THIS_MODULE<span class="token punctuation">,</span>
	<span class="token punctuation">.</span>unlocked_ioctl	<span class="token operator">&#61;</span> exporter_ioctl<span class="token punctuation">,</span>
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
<p>在 ioctl 中&#xff0c;通过 <em>dma_buf_fd()</em> 创建一个新的 fd&#xff0c;并通过 <em>copy_to_user()</em> 将该 fd 的值传给上层应用程序。</p> 
<h4><a id="importer__135"></a>importer 驱动</h4> 
<p>我们基于<a href="https://blog.csdn.net/hexiaolong2009/article/details/102596761#t4">《dma-buf 由浅入深&#xff08;二&#xff09; —— kmap/vmap》</a>中的 importer-kmap.c 进行修改。</p> 
<p><a href="https://github.com/hexiaolong2008/sample-code/blob/master/dma-buf/06/importer-fd.c">importer-fd.c</a></p> 
<pre><code class="prism language-c"><span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/dma-buf.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/module.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/miscdevice.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/uaccess.h&gt;</span></span>

<span class="token keyword">static</span> <span class="token keyword">int</span> <span class="token function">importer_test</span><span class="token punctuation">(</span><span class="token keyword">struct</span> dma_buf <span class="token operator">*</span>dmabuf<span class="token punctuation">)</span>
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

<span class="token keyword">static</span> <span class="token keyword">long</span> <span class="token function">importer_ioctl</span><span class="token punctuation">(</span><span class="token keyword">struct</span> file <span class="token operator">*</span>filp<span class="token punctuation">,</span> <span class="token keyword">unsigned</span> <span class="token keyword">int</span> cmd<span class="token punctuation">,</span> <span class="token keyword">unsigned</span> <span class="token keyword">long</span> arg<span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token keyword">int</span> fd<span class="token punctuation">;</span>
	<span class="token keyword">struct</span> dma_buf <span class="token operator">*</span>dmabuf<span class="token punctuation">;</span>

	<span class="token function">copy_from_user</span><span class="token punctuation">(</span><span class="token operator">&amp;</span>fd<span class="token punctuation">,</span> <span class="token punctuation">(</span><span class="token keyword">void</span> __user <span class="token operator">*</span><span class="token punctuation">)</span>arg<span class="token punctuation">,</span> <span class="token keyword">sizeof</span><span class="token punctuation">(</span><span class="token keyword">int</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">;</span>

	dmabuf <span class="token operator">&#61;</span> <span class="token function">dma_buf_get</span><span class="token punctuation">(</span>fd<span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token function">importer_test</span><span class="token punctuation">(</span>dmabuf<span class="token punctuation">)</span><span class="token punctuation">;</span>
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
<p>与 importer-kmap 驱动相比&#xff0c;上面的驱动新增了 misc driver 部分&#xff0c;通过 <em>ioctl</em> 接口来接收上层传下来的 fd&#xff0c;并通过 <em>dma_buf_get()</em> 将 fd 转换成 dma-buf 指针。随后便在 kernel 空间通过 kmap/vmap 来访问该 dma-buf 的物理内存。</p> 
<p>需要注意的是&#xff0c;<em>dma_buf_get()</em> 会增加 dma-buf 的引用计数&#xff0c;所以在使用完 dma-buf 后&#xff0c;要记得用 <em>dma_buf_put()</em> 将引用计数再减回来&#xff0c;否则引用计数不匹配&#xff0c;将导致 dma-buf 的 <em>release</em> 接口无法被回调&#xff0c;从而导致 buffer 无法被释放&#xff0c;造成内存泄漏。</p> 
<h4><a id="userspace__203"></a>userspace 程序</h4> 
<p><a href="https://github.com/hexiaolong2008/sample-code/blob/master/dma-buf/06/dmabuf-test/share_fd.c">share_fd.c</a></p> 
<pre><code class="prism language-c"><span class="token keyword">int</span> <span class="token function">main</span><span class="token punctuation">(</span><span class="token keyword">int</span> argc<span class="token punctuation">,</span> <span class="token keyword">char</span> <span class="token operator">*</span>argv<span class="token punctuation">[</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token keyword">int</span> fd<span class="token punctuation">;</span>
	<span class="token keyword">int</span> dmabuf_fd <span class="token operator">&#61;</span> <span class="token number">0</span><span class="token punctuation">;</span>

	fd <span class="token operator">&#61;</span> <span class="token function">open</span><span class="token punctuation">(</span><span class="token string">&#34;/dev/exporter&#34;</span><span class="token punctuation">,</span> O_RDONLY<span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token function">ioctl</span><span class="token punctuation">(</span>fd<span class="token punctuation">,</span> <span class="token number">0</span><span class="token punctuation">,</span> <span class="token operator">&amp;</span>dmabuf_fd<span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token function">close</span><span class="token punctuation">(</span>fd<span class="token punctuation">)</span><span class="token punctuation">;</span>

	fd <span class="token operator">&#61;</span> <span class="token function">open</span><span class="token punctuation">(</span><span class="token string">&#34;/dev/importer&#34;</span><span class="token punctuation">,</span> O_RDONLY<span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token function">ioctl</span><span class="token punctuation">(</span>fd<span class="token punctuation">,</span> <span class="token number">0</span><span class="token punctuation">,</span> <span class="token operator">&amp;</span>dmabuf_fd<span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token function">close</span><span class="token punctuation">(</span>fd<span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token keyword">return</span> <span class="token number">0</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>
</code></pre> 
<p>该应用程序做的事情很简单&#xff0c;就是将 dma-buf 的 fd 从 exporter 传递给 importer 驱动。这里为了简单起见&#xff0c;ioctl() 第二个参数没有任何意义&#xff0c;可以忽略。</p> 
<h3><a id="_225"></a>开发环境</h3> 
<table><thead><tr><th align="left"></th><th align="left"></th></tr></thead><tbody><tr><td align="left">内核源码</td><td align="left"><a href="https://mirrors.edge.kernel.org/pub/linux/kernel/v4.x/linux-4.14.143.tar.xz" rel="nofollow">4.14.143</a></td></tr><tr><td align="left">示例源码</td><td align="left"><a href="https://github.com/hexiaolong2008/sample-code/tree/master/dma-buf/06">hexiaolong2008-GitHub/sample-code/dma-buf/06</a></td></tr><tr><td align="left">开发平台</td><td align="left">Ubuntu14.04/16.04</td></tr><tr><td align="left">运行平台</td><td align="left"><a href="https://github.com/hexiaolong2008/my-qemu">my-qemu 仿真环境</a></td></tr></tbody></table>
<h3><a id="_232"></a>运行</h3> 
<p>在 my-qemu 仿真环境中执行如下命令&#xff1a;</p> 
<pre><code class="prism language-handlebars"><span class="token punctuation">#</span> <span class="token variable">insmod</span> <span class="token punctuation">/</span><span class="token variable">lib</span><span class="token punctuation">/</span><span class="token variable">modules</span><span class="token punctuation">/</span><span class="token number">4.14</span><span class="token number">.143</span><span class="token block keyword">/kernel/drivers/dma-buf/importer-fd.ko</span>
<span class="token punctuation">#</span> <span class="token variable">insmod</span> <span class="token punctuation">/</span><span class="token variable">lib</span><span class="token punctuation">/</span><span class="token variable">modules</span><span class="token punctuation">/</span><span class="token number">4.14</span><span class="token number">.143</span><span class="token block keyword">/kernel/drivers/dma-buf/exporter-fd.ko</span>
<span class="token punctuation">#</span> <span class="token punctuation">.</span><span class="token punctuation">/</span><span class="token variable">share_fd</span>
</code></pre> 
<p>将看到如下打印结果&#xff1a;</p> 
<pre><code>read from dmabuf kmap: hello world!
read from dmabuf vmap: hello world!
</code></pre> 
<p>通过上面的运行结果我们看到&#xff0c;即使 importer 驱动先加载&#xff0c;也不会影响应用程序的输出结果&#xff0c;真正实现了 importer 驱动与 exporter 驱动之间的解耦合。</p> 
<h3><a id="_fd_247"></a>跨进程 fd</h3> 
<p>做 Linux 应用开发的同事都知道&#xff0c;fd 属于进程资源&#xff0c;它的作用域只在单个进程空间范围内有效&#xff0c;<strong>即同样的 fd 值&#xff0c;在进程 A 和 进程 B 中所指向的文件是不同的</strong>。因此 fd 是不能在多个进程之间共享的&#xff0c;也就是说 <code>dma_buf_fd()</code> 与 <code>dma_buf_get()</code> 只能是在同一进程中调用。</p> 
<p>但是有的小伙伴就会问了&#xff1a;在 Android 系统中&#xff0c;dma-buf 几乎都是由 ION 来统一分配的&#xff0c;ION 所在进程&#xff08;Allocator&#xff09;在分配好 buffer 以后&#xff0c;会将该 buffer 所对应的 fd 传给其它进程&#xff0c;如 SurfaceFlinger 或 CameraService&#xff0c;而这些进程在收到 fd 后在各自的底层驱动中都能正确的转换成相应的 dma-buf&#xff0c;那这又是如何做到的呢&#xff1f;</p> 
<p>fd 并不是完全不能在多进程中共享&#xff0c;而是需要采用特殊的方式进行传递。在 linux 系统中&#xff0c;最常用的做法就是通过 socket 来实现 fd 的传递。而在 Android 系统中&#xff0c;则是通过 Binder 来实现的。需要注意的是&#xff0c;传递后 fd 的值可能会发生变化&#xff0c;但是它们所指向的文件都是同一文件。关于 Binder 如何实现 fd 跨进程共享&#xff0c;请见<a href="https://blog.csdn.net/hexiaolong2009/article/details/102596802#t12">参考资料</a>中的第一篇文章&#xff0c;这里不做赘述。总之&#xff0c;有了 Binder&#xff0c;<em>dma_buf_fd()</em> 和 <em>dma_buf_get()</em> 就可以不用严格限制在同一进程中使用了。</p> 
<h3><a id="_254"></a>总结</h3> 
<ul><li><strong>为什么需要 fd &#xff1f;</strong></li></ul> 
<ol><li>方便应用程序直接在 user space 访问该 buffer&#xff08;通过 mmap&#xff09;&#xff1b;</li><li>方便该 buffer 在各个驱动模块之间流转&#xff0c;而无需拷贝&#xff1b;</li><li>降低了各驱动之间的耦合度&#xff1b;</li></ol> 
<ul><li><strong>如何实现 fd 跨进程共享&#xff1f;</strong> Binder!</li><li><strong>get / put 将影响 dma-buf 的内存释放</strong></li></ul> 
<h3><a id="_264"></a>参考资料</h3> 
<ol><li><a href="https://blog.csdn.net/zhangjg_blog/article/details/83502195">Android Binder传递文件描述符原理分析</a></li><li><a href="https://blog.csdn.net/majianfei1023/article/details/51454797">进程间传递文件描述符–sendmsg,recvmsg</a></li></ol> 
<br /> 
<br /> 
<br /> 
<p>上一篇&#xff1a;<a href="https://blog.csdn.net/hexiaolong2009/article/details/102596791">《dma-buf 由浅入深&#xff08;四&#xff09;—— mmap》</a><br /> 下一篇&#xff1a;<a href="https://blog.csdn.net/hexiaolong2009/article/details/102596825">《dma-buf 由浅入深&#xff08;六&#xff09;—— begin / end cpu_access》</a><br /> 文章汇总&#xff1a;<a href="https://blog.csdn.net/hexiaolong2009/article/details/83720940">《DRM&#xff08;Direct Rendering Manager&#xff09;学习简介》</a></p>
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
                    22
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
                <span class="count get-collection " data-num="63" id="get-collection">
                    63
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
                    5
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
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/shenjunpeng/article/details/151726143"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6661.1","mod":"popu_871","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant_t0.none-task-blog-2~default~BlogCommendFromBaidu~PaidSort-1-151726143-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"1","strategy":"2~default~BlogCommendFromBaidu~PaidSort","dest":"https://blog.csdn.net/shenjunpeng/article/details/151726143"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/shenjunpeng/article/details/151726143" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6661.1","mod":"popu_871","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant_t0.none-task-blog-2~default~BlogCommendFromBaidu~PaidSort-1-151726143-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"1","strategy":"2~default~BlogCommendFromBaidu~PaidSort","dest":"https://blog.csdn.net/shenjunpeng/article/details/151726143"}'  data-report-query='spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7EPaidSort-1-151726143-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7EPaidSort-1-151726143-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">第<em>五</em>章：BO的共享：5.2.3 <em>dma</em><em>-</em><em>buf</em>的实现</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/shenjunpeng" target="_blank"><span class="blog-title">deeplyMind</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">09-15</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					1102
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/shenjunpeng/article/details/151726143" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6661.1","mod":"popu_871","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant_t0.none-task-blog-2~default~BlogCommendFromBaidu~PaidSort-1-151726143-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"1","strategy":"2~default~BlogCommendFromBaidu~PaidSort","dest":"https://blog.csdn.net/shenjunpeng/article/details/151726143"}'  data-report-query='spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7EPaidSort-1-151726143-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7EPaidSort-1-151726143-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">本文介绍了Linux内核中<em>DMA</em><em>-</em><em>BUF</em>共享内存框架的原理与实现。该框架通过文件描述符<em>(</em><em>fd</em><em>)</em>作为统一接口，实现跨设备、跨驱动的物理内存共享。核心流程包括导出者<em>(</em>Exporter<em>)</em>分配缓冲区并创建<em>dma</em>_<em>buf</em>结构返回<em>fd</em>，导入者<em>(</em>Importer<em>)</em>通过<em>fd</em>获取缓冲区进行<em>DMA</em>操作。框架采用引用计数管理生命周期，通过<em>dma</em>_resv和fence机制保证同步安全。文章详细分析了<em>dma</em>_<em>buf</em>_export、<em>dma</em>_<em>buf</em>_get等核心函数的实现，阐述了其&quot;一切皆文件&quot;的设计哲学。</div>
			</a>
		</div>
	</div>
</div>
                </div>
            <script src="https://csdnimg.cn/release/blogv2/dist/components/js/pc_wap_commontools-829a4838ae.min.js" type="text/javascript" async></script>
              <div class="second-recommend-box recommend-box ">
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/relax33/article/details/128319124"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.1","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~OPENSEARCH~PaidSort-1-128319124-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"1","strategy":"2~default~OPENSEARCH~PaidSort","dest":"https://blog.csdn.net/relax33/article/details/128319124"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/relax33/article/details/128319124" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.1","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~OPENSEARCH~PaidSort-1-128319124-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"1","strategy":"2~default~OPENSEARCH~PaidSort","dest":"https://blog.csdn.net/relax33/article/details/128319124"}'  data-report-query='spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7EPaidSort-1-128319124-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7EPaidSort-1-128319124-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
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
			<a href="https://blog.csdn.net/relax33/article/details/128319124" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.1","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~OPENSEARCH~PaidSort-1-128319124-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"1","strategy":"2~default~OPENSEARCH~PaidSort","dest":"https://blog.csdn.net/relax33/article/details/128319124"}'  data-report-query='spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7EPaidSort-1-128319124-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7EPaidSort-1-128319124-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">ION <em>DMA</em><em>-</em><em>BUF</em> IOMMU</div>
			</a>
		</div>
	</div>
</div>
              </div>
<a id="commentBox" name="commentBox"></a>
  <div id="pcCommentBox" class="comment-box comment-box-new2 unlogin-comment-box-new" style="display:none">
      <div class="unlogin-comment-model">
          <span class="unlogin-comment-tit">5&nbsp;条评论</span>
        <span class="unlogin-comment-text">您还未登录，请先</span>
        <span class="unlogin-comment-bt">登录</span>
        <span class="unlogin-comment-text">后发表或查看评论</span>
      </div>
  </div>
  <div class="blog-comment-box-new" style="display: none;">
        <h1>5 条评论</h1>
      <ul>
          <li>
            <a target="_blank" href="https://blog.csdn.net/weixin_41954817">
              <img src="https://profile-avatar.csdnimg.cn/default.jpg!1"
                alt="weixin_41954817" class="avatar">
            </a>
            <div class="right-box">
              <div class="new-info-box clearfix">
                <div class="comment-top">
                  <div class="user-box">
                    <a class="name-href" target="_blank"  href="https://blog.csdn.net/weixin_41954817">
                      <span class="name ">weixin_41954817</span></a>
                    <span class="date" title="2021-07-23 15:19:33">2021.07.23</span>
                    <div class="new-comment">博主你好，我认为将exporter_alloc_page() 放到 exporter_init()中不妥，因为当进程退出的时候，dma_buf_fd所对应的fd文件也将会被关闭，而这个文件是dma_buf的结构体中存储的，在结束时会被释放，这样一来第二次运行进程share_fd时dma_buf_get将找不到fd对应的文件。事实上用你的demo测试一下会发现，连续运行两次share_fd只会打印一次hello world,而第二次运行时会从dma_buf_get 返回错误然后退出。</div>
                  </div>
                </div>
              </div>
            </div>
          </li>
          <li >
            <ul>
                <li>
                  <a target="_blank" href="https://blog.csdn.net/czg13548930186">
                    <img src="https://profile-avatar.csdnimg.cn/4c7df85a4f414b95bc0e7900c831fa10_czg13548930186.jpg!1"
                      alt="czg13548930186" class="avatar">
                  </a>
                  <div class="right-box">
                    <div class="new-info-box clearfix">
                      <div class="comment-top">
                        <div class="user-box">
                          <a class="name-href" target="_blank"  href="https://blog.csdn.net/czg13548930186">
                            <span class="name ">种瓜大爷</span><span class="text">回复</span><span class="nick-name">weixin_41954817</span>
                          </a>
                          <span class="date" title="2022-01-14 14:06:05">2022.01.14</span>
                          <div class="new-comment">应该是方便教程，正常使用不会这样</div>
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
            <a target="_blank" href="https://blog.csdn.net/yuhaojin">
              <img src="https://profile-avatar.csdnimg.cn/673ff47616774da09243a519d2b3b3fe_yuhaojin.jpg!1"
                alt="yuhaojin" class="avatar">
            </a>
            <div class="right-box">
              <div class="new-info-box clearfix">
                <div class="comment-top">
                  <div class="user-box">
                    <a class="name-href" target="_blank"  href="https://blog.csdn.net/yuhaojin">
                      <span class="name ">yuhaojin</span></a>
                    <span class="date" title="2021-07-22 20:40:13">2021.07.22</span>
                    <div class="new-comment">如果kernel里面一直put的话，应该是会先调用dma-buf ops的release释放实际的物理内存空间和ion_buffer。等到后面close fd的时候，会调用file_opreation的release，释放struct dma_buf，逻辑是这样的吧？</div>
                  </div>
                </div>
              </div>
            </div>
          </li>
      </ul>
      <ul>
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
                      <span class="name ">何小龙</span></a>
                    <span class="date" title="2020-08-12 13:32:34">2020.08.12</span>
                    <div class="new-comment">对啊，只要应用程序不打算再使用该dmabuf了，就应该主动close。如果你的应用程序只是one shot型，即运行一下就自动退出了，那么可以不主动close，因为进程退出时，操作系统会自动帮你释放当前进程所有已申请的进程资源（包括fd）。</div>
                  </div>
                </div>
              </div>
            </div>
          </li>
      </ul>
      <ul>
          <li>
            <a target="_blank" href="https://blog.csdn.net/ceiba2002">
              <img src="https://profile-avatar.csdnimg.cn/89b2128a7b1a4c1c8ecea74d19d351ca_ceiba2002.jpg!1"
                alt="ceiba2002" class="avatar">
            </a>
            <div class="right-box">
              <div class="new-info-box clearfix">
                <div class="comment-top">
                  <div class="user-box">
                    <a class="name-href" target="_blank"  href="https://blog.csdn.net/ceiba2002">
                      <span class="name ">依云听风</span></a>
                    <span class="date" title="2020-08-11 13:48:50">2020.08.11</span>
                    <div class="new-comment">userspace得到dmabuf_fd是不是也需要在适当时候close(dmabuf_fd) ?</div>
                  </div>
                </div>
              </div>
            </div>
          </li>
      </ul>
    </div>
              <div class="recommend-box insert-baidu-box recommend-box-style ">
                <div class="recommend-item-box no-index" style="display:none"></div>
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/qq_28637193/article/details/106963263"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.2","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~Ctr-2-106963263-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"2","strategy":"2~default~baidujs_baidulandingword~Ctr","dest":"https://blog.csdn.net/qq_28637193/article/details/106963263"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/qq_28637193/article/details/106963263" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.2","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~Ctr-2-106963263-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"2","strategy":"2~default~baidujs_baidulandingword~Ctr","dest":"https://blog.csdn.net/qq_28637193/article/details/106963263"}'  data-report-query='spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7Ebaidujs_baidulandingword%7ECtr-2-106963263-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7Ebaidujs_baidulandingword%7ECtr-2-106963263-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">以高通camera 申请ion内存看<em>dma</em><em>-</em><em>buf</em></div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/qq_28637193" target="_blank"><span class="blog-title">兔兔里个花兔的博客</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">06-26</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					6243
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/qq_28637193/article/details/106963263" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.2","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~Ctr-2-106963263-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"2","strategy":"2~default~baidujs_baidulandingword~Ctr","dest":"https://blog.csdn.net/qq_28637193/article/details/106963263"}'  data-report-query='spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7Ebaidujs_baidulandingword%7ECtr-2-106963263-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7Ebaidujs_baidulandingword%7ECtr-2-106963263-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">1 <em>fd</em> 与ion <em>buf</em>fer， <em>file</em>绑定

 简单总结就是申请一个<em>buf</em>fer，再创建一个<em>dma</em><em>buf</em> 结构体然后将，然后将<em>dma</em><em>buf</em>中得指针指向<em>buf</em>fer，<em>dma</em><em>buf</em> 再传递给一个匿名的inode，获取到一个<em>file</em>，这样<em>file</em>和<em>dma</em><em>buf</em>绑定起来，也就和<em>buf</em>fer关联上。然后再从进程中分配一个空闲的<em>fd</em>，将<em>fd</em> 和<em>file</em>囊绑定。这样就能通过<em>fd</em> 快速查找到<em>buf</em>fer。<em>file</em> 是个全系统的，他和进程无关，但是<em>fd</em> 是每个进程都是自己独立的，所以再跨进程<em>传输</em>时只需要保证<em>fd</em> ...</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/fuchangyaocool/article/details/147563638"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.3","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~Ctr-3-147563638-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"3","strategy":"2~default~baidujs_baidulandingword~Ctr","dest":"https://blog.csdn.net/fuchangyaocool/article/details/147563638"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/fuchangyaocool/article/details/147563638" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.3","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~Ctr-3-147563638-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"3","strategy":"2~default~baidujs_baidulandingword~Ctr","dest":"https://blog.csdn.net/fuchangyaocool/article/details/147563638"}'  data-report-query='spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7Ebaidujs_baidulandingword%7ECtr-3-147563638-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7Ebaidujs_baidulandingword%7ECtr-3-147563638-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">Linux <em>dma</em><em>buf</em>机制详解</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/fuchangyaocool" target="_blank"><span class="blog-title">fuchangyaocool的博客</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">04-27</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					1492
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/fuchangyaocool/article/details/147563638" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.3","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~Ctr-3-147563638-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"3","strategy":"2~default~baidujs_baidulandingword~Ctr","dest":"https://blog.csdn.net/fuchangyaocool/article/details/147563638"}'  data-report-query='spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7Ebaidujs_baidulandingword%7ECtr-3-147563638-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7Ebaidujs_baidulandingword%7ECtr-3-147563638-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">（分离<em>DMA</em>缓冲区）指的是解除某个设备或进程与已附加（attached）的<em>DMA</em>缓冲区的关联关系。设备或驱动通过附加（attach）操作获取对<em>DMA</em>缓冲区的访问权限，可能涉及内存映射（mapping）或地址转换（如IOMMU操作）。当设备不再需要访问缓冲区时，通过分离（detach）操作释放与缓冲区的关联，撤销之前的映射或资源占用。：基于<em>DMA</em>的缓冲区抽象，提供跨设备/子系统的内存共享，避免数据在用户空间和内核间的冗余拷贝。确保设备写入内存的数据对CPU可见，需使CPU缓存失效（Invalidate）。</div>
			</a>
		</div>
	</div>
</div>
		<dl id="recommend-item-box-tow" class="recommend-item-box type_blog clearfix">
			
		</dl>
<div class="recommend-item-box type_blog clearfix" data-url="https://zhugeyifan.blog.csdn.net/article/details/154238790"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.4","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~YuanLiJiHua~Position-4-154238790-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"4","strategy":"2~default~YuanLiJiHua~Position","dest":"https://zhugeyifan.blog.csdn.net/article/details/154238790"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://zhugeyifan.blog.csdn.net/article/details/154238790" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.4","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~YuanLiJiHua~Position-4-154238790-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"4","strategy":"2~default~YuanLiJiHua~Position","dest":"https://zhugeyifan.blog.csdn.net/article/details/154238790"}'  data-report-query='spm=1001.2101.3001.6650.4&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-4-154238790-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-4-154238790-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">【<em>内存管理</em>】深入理解内存映射（Memory Mapping）与mmap：实现高效零拷贝的<em>DMA</em>缓冲区共享</div>
					<div class="tag">最新发布</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/Ivan804638781" target="_blank"><span class="blog-title">诸葛一帆丶的博客</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">11-01</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					1523
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://zhugeyifan.blog.csdn.net/article/details/154238790" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.4","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~YuanLiJiHua~Position-4-154238790-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"4","strategy":"2~default~YuanLiJiHua~Position","dest":"https://zhugeyifan.blog.csdn.net/article/details/154238790"}'  data-report-query='spm=1001.2101.3001.6650.4&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-4-154238790-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-4-154238790-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">内存映射<em>(</em>mmap<em>)</em>是操作系统提供的将文件或设备直接映射到进程虚拟地址空间的机制，允许进程像访问内存一样访问文件内容或设备内存区域。文章详细介绍了mmap系统调用的工作机制，重点阐述了将内核<em>DMA</em>缓冲区映射到用户空间的完整流程，包括驱动端<em>DMA</em>缓冲区分配、mmap文件操作实现以及用户空间映射方法。通过remap_pfn_range或<em>dma</em>_mmap_coherent函数建立物理页到用户空间的映射，实现零拷贝数据<em>传输</em>。相比传统read/write方式，mmap消除了数据拷贝开销，减少了系统调用次数，提升了性能</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/IT_Beijing_BIT/article/details/140369236"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.5","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-5-140369236-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"5","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/IT_Beijing_BIT/article/details/140369236"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/IT_Beijing_BIT/article/details/140369236" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.5","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-5-140369236-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"5","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/IT_Beijing_BIT/article/details/140369236"}'  data-report-query='spm=1001.2101.3001.6650.5&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-5-140369236-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-5-140369236-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">缓冲区共享和同步<em>dma</em>_<em>buf</em> 之一</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/IT_Beijing_BIT" target="_blank"><span class="blog-title">IT_Beijing_BIT的博客</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">07-13</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					2323
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/IT_Beijing_BIT/article/details/140369236" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.5","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-5-140369236-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"5","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/IT_Beijing_BIT/article/details/140369236"}'  data-report-query='spm=1001.2101.3001.6650.5&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-5-140369236-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-5-140369236-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1"><em>dma</em><em>-</em><em>buf</em> 子系统提供了用于跨多个设备驱动程序和子系统共享硬件 <em>(</em><em>DMA</em><em>)</em> 访问缓冲区以及同步异步硬件访问的框架。</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/huifeidedabian/article/details/115659723"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.6","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-6-115659723-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"6","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/huifeidedabian/article/details/115659723"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/huifeidedabian/article/details/115659723" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.6","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-6-115659723-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"6","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/huifeidedabian/article/details/115659723"}'  data-report-query='spm=1001.2101.3001.6650.6&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-6-115659723-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-6-115659723-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">v4l2 use V4L2_MEMORY_MMAP方式导出为 <em>DMA</em> <em>BUF</em> <em>fd</em>  方式使用</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/huifeidedabian" target="_blank"><span class="blog-title">huifeidedabian的博客</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">04-13</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					5145
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/huifeidedabian/article/details/115659723" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.6","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-6-115659723-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"6","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/huifeidedabian/article/details/115659723"}'  data-report-query='spm=1001.2101.3001.6650.6&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-6-115659723-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-6-115659723-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">V4L2_MEMORY_MMAP 导出 <em>fd</em> 需要使用  vb2_ioctl_exp<em>buf</em>  <em>(</em>只能使用于VB2_MEMORY_MMAP 方式<em>)</em>。
int <em>buf</em>fer_export<em>(</em>int v4l<em>fd</em>, enum v4l2_<em>buf</em>_type bt, int index, int *<em>dma</em><em>fd</em><em>)</em>
{
    struct v4l2_export<em>buf</em>fer exp<em>buf</em>;

    memset<em>(</em>&amp;exp<em>buf</em>, 0, sizeof<em>(</em>exp<em>buf</em><em>)</em><em>)</em>;
    exp<em>buf</em>.type = bt;
 </div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://devpress.csdn.net/v1/article/detail/7940330"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.7","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-7-7940330-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"7","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/7940330"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://devpress.csdn.net/v1/article/detail/7940330" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.7","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-7-7940330-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"7","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/7940330"}'  data-report-query='spm=1001.2101.3001.6650.7&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-7-7940330-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-7-7940330-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">linux之<em>DMA</em><em>-</em><em>BUF</em> API使用指南</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/crazyjiang" target="_blank"><span class="blog-title">crazyjiang的专栏</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">09-04</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					2万+
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://devpress.csdn.net/v1/article/detail/7940330" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.7","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-7-7940330-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"7","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/7940330"}'  data-report-query='spm=1001.2101.3001.6650.7&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-7-7940330-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-7-7940330-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1"><em>DMA</em><em>-</em><em>BUF</em> API使用指南
by JHJ<em>(</em>jianghuijun211@gmail.com<em>)</em>
转载出自：http://blog.csdn.net/crazyjiang
本文将会告诉驱动开发者什么是<em>dma</em><em>-</em><em>buf</em>共享缓冲区接口，如何作为一个生产者及消费者使用共享缓冲区。
任何一个设备驱动想要使用<em>DMA</em>共享缓冲区，就必须为缓冲区的生产者或者消费者。
如果驱动A想用驱动B创建的缓冲区，那么</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/qq_18998145/article/details/99406944"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.8","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-8-99406944-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"8","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/qq_18998145/article/details/99406944"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/qq_18998145/article/details/99406944" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.8","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-8-99406944-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"8","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/qq_18998145/article/details/99406944"}'  data-report-query='spm=1001.2101.3001.6650.8&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-8-99406944-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-8-99406944-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1"><em>DMA</em>_<em>BUF</em></div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/qq_18998145" target="_blank"><span class="blog-title">LIEY</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">08-13</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					2990
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/qq_18998145/article/details/99406944" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.8","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-8-99406944-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"8","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/qq_18998145/article/details/99406944"}'  data-report-query='spm=1001.2101.3001.6650.8&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-8-99406944-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-8-99406944-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">考虑这样一种场景，摄像头采集的视频数据需要送到GPU中进行编码、显示。负责数据采集和编码的模块是Linux下不同的驱动设备，将采集设备中的数据送到编码设备中 需要一种方法。最简单的方法可能就是进行一次内存拷贝，但是我们这里需要寻求一种免拷贝的通用方法。<em>dma</em>_<em>buf</em>是内核中一个独立的子系统，可以让不同设备、子系统之间进行内存共享的统一机制。

<em>DMA</em>_<em>BUF</em>框架下主要有两个角色对象，一个是expo...</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_chatgpt clearfix" data-url="https://wenku.csdn.net/answer/2qr6q87w5b"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.9","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Rate-11-1c943n8fp3-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"11","strategy":"2~default~OPENSEARCH~Rate","dest":"https://wenku.csdn.net/answer/2qr6q87w5b"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://wenku.csdn.net/answer/2qr6q87w5b" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.9","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Rate-11-1c943n8fp3-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"11","strategy":"2~default~OPENSEARCH~Rate","dest":"https://wenku.csdn.net/answer/2qr6q87w5b"}'  data-report-query='spm=1001.2101.3001.6650.9&utm_medium=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7ERate-11-1c943n8fp3-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7ERate-11-1c943n8fp3-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
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
			<a href="https://wenku.csdn.net/answer/2qr6q87w5b" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.9","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Rate-11-1c943n8fp3-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"11","strategy":"2~default~OPENSEARCH~Rate","dest":"https://wenku.csdn.net/answer/2qr6q87w5b"}'  data-report-query='spm=1001.2101.3001.6650.9&utm_medium=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7ERate-11-1c943n8fp3-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7ERate-11-1c943n8fp3-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1"> 成功导出后，可通过 `<em>file</em><em>-</em>&gt;f_dentry<em>-</em>&gt;d_inode<em>-</em>&gt;i_cdev` 得到对应的字符设备节点，并将其转换为文件描述符传递给应用程序。  <em>-</em><em>-</em><em>-</em>  #### 进程间通信 当另一个进程接收到此文件描述符时，它可以利用以下 API 加载缓冲...</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_chatgpt clearfix" data-url="https://wenku.csdn.net/answer/3xk6xy573a"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.10","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Rate-11-1c943n8fp3-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"11","strategy":"2~default~OPENSEARCH~Rate","dest":"https://wenku.csdn.net/answer/3xk6xy573a"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://wenku.csdn.net/answer/3xk6xy573a" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.10","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Rate-11-1c943n8fp3-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"11","strategy":"2~default~OPENSEARCH~Rate","dest":"https://wenku.csdn.net/answer/3xk6xy573a"}'  data-report-query='spm=1001.2101.3001.6650.10&utm_medium=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7ERate-11-1c943n8fp3-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7ERate-11-1c943n8fp3-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1"><em>dma</em>_<em>buf</em>的源码在哪个位置</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info display-flex">
					<span class="info-block">10-31</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://wenku.csdn.net/answer/3xk6xy573a" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.10","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Rate-11-1c943n8fp3-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"11","strategy":"2~default~OPENSEARCH~Rate","dest":"https://wenku.csdn.net/answer/3xk6xy573a"}'  data-report-query='spm=1001.2101.3001.6650.10&utm_medium=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7ERate-11-1c943n8fp3-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7ERate-11-1c943n8fp3-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">我们被要求查找<em>dma</em>_<em>buf</em>源码在Linux内核中的位置。根据Linux内核源码的结构，<em>dma</em>_<em>buf</em>相关...[^1]: <em>dma</em><em>-</em><em>buf</em> <em>由浅入深</em>（四） &mdash;&mdash; mmap [^2]: <em>dma</em>_<em>buf</em>学习记录之一基础知识 [^3]: linux 之<em>dma</em>_<em>buf</em> <em>(</em>1<em>)</em><em>-</em> <em>dma</em>_<em>buf</em> 的初步介绍</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_chatgpt clearfix" data-url="https://wenku.csdn.net/answer/1c943n8fp3"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.11","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Rate-11-1c943n8fp3-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"11","strategy":"2~default~OPENSEARCH~Rate","dest":"https://wenku.csdn.net/answer/1c943n8fp3"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://wenku.csdn.net/answer/1c943n8fp3" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.11","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Rate-11-1c943n8fp3-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"11","strategy":"2~default~OPENSEARCH~Rate","dest":"https://wenku.csdn.net/answer/1c943n8fp3"}'  data-report-query='spm=1001.2101.3001.6650.11&utm_medium=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7ERate-11-1c943n8fp3-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7ERate-11-1c943n8fp3-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1"><em>BUF</em> PUT B2 I1 J1 &quot;Cell 1&quot; <em>-</em>0.8e<em>-</em>2 &quot;Cell 3&quot; &quot;Cell 4&quot;</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info display-flex">
					<span class="info-block">09-23</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://wenku.csdn.net/answer/1c943n8fp3" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.11","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Rate-11-1c943n8fp3-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"11","strategy":"2~default~OPENSEARCH~Rate","dest":"https://wenku.csdn.net/answer/1c943n8fp3"}'  data-report-query='spm=1001.2101.3001.6650.11&utm_medium=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7ERate-11-1c943n8fp3-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7ERate-11-1c943n8fp3-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">如果你需要更多关于<em>dma</em><em>-</em><em>buf</em>的信息，我建议你参考《<em>dma</em><em>-</em><em>buf</em> <em>由浅入深</em>》系列文章中的相关章节，特别是《<em>dma</em><em>-</em><em>buf</em> <em>由浅入深</em>（<em>五</em>）&mdash;&mdash; <em>File</em>》章节。这些文章将会对<em>dma</em><em>-</em><em>buf</em>的基本概念和使用进行详细解释。</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/langjian2012/article/details/144420600"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.12","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~OPENSEARCH~Rate-12-144420600-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"12","strategy":"2~default~OPENSEARCH~Rate","dest":"https://blog.csdn.net/langjian2012/article/details/144420600"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/langjian2012/article/details/144420600" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.12","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~OPENSEARCH~Rate-12-144420600-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"12","strategy":"2~default~OPENSEARCH~Rate","dest":"https://blog.csdn.net/langjian2012/article/details/144420600"}'  data-report-query='spm=1001.2101.3001.6650.12&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7ERate-12-144420600-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7ERate-12-144420600-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">memory内存分类</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/langjian2012" target="_blank"><span class="blog-title">langjian2012的专栏</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">12-12</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					474
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/langjian2012/article/details/144420600" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.12","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~OPENSEARCH~Rate-12-144420600-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"12","strategy":"2~default~OPENSEARCH~Rate","dest":"https://blog.csdn.net/langjian2012/article/details/144420600"}'  data-report-query='spm=1001.2101.3001.6650.12&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7ERate-12-144420600-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7ERate-12-144420600-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1"><em>(</em>adb shell dumpsys meminfo x<em>)</em>堆内存用于存储对象实例和静态变量</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/fs3296/article/details/125387687"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.13","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-13-125387687-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"13","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/fs3296/article/details/125387687"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/fs3296/article/details/125387687" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.13","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-13-125387687-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"13","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/fs3296/article/details/125387687"}'  data-report-query='spm=1001.2101.3001.6650.13&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-13-125387687-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-13-125387687-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">Linux图形子系统之<em>dma</em><em>-</em><em>buf</em></div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/fs3296" target="_blank"><span class="blog-title">fs3296的博客</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">06-21</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					2997
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/fs3296/article/details/125387687" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.13","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-13-125387687-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"13","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/fs3296/article/details/125387687"}'  data-report-query='spm=1001.2101.3001.6650.13&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-13-125387687-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-13-125387687-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1"><em>dma</em><em>-</em><em>buf</em>是linux内核提供的一种机制，用于不同模块实现内存共享。它提供生产者和消费者模式来实现不同模块对内存共享同时，不用关心各个模块的内部实现细节，从而解耦。在<em>drm</em>框架中也集成了<em>dma</em><em>-</em><em>buf</em>方式的<em>内存管理</em>。<em>drm</em>通过<em>DRM</em>_IOCTL_PRIME_HANDLE_TO_<em>FD</em>实现将一个gem对象句柄转为<em>dma</em><em>-</em><em>buf</em>的<em>fd</em>。其中会调用struct <em>drm</em>_driver的prime_handle_to_<em>fd</em>回调，<em>drm</em>_gem_prime_handle_to_<em>fd</em>函数是prime_handle_to</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/weixin_39592381/article/details/133790159"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.14","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-14-133790159-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"14","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/weixin_39592381/article/details/133790159"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/weixin_39592381/article/details/133790159" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.14","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-14-133790159-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"14","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/weixin_39592381/article/details/133790159"}'  data-report-query='spm=1001.2101.3001.6650.14&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-14-133790159-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-14-133790159-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1"><em>DMA</em><em>-</em><em>BUF</em></div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/weixin_39592381" target="_blank"><span class="blog-title">weixin_39592381的博客</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">10-13</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					752
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/weixin_39592381/article/details/133790159" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.14","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-14-133790159-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"14","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/weixin_39592381/article/details/133790159"}'  data-report-query='spm=1001.2101.3001.6650.14&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-14-133790159-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-14-133790159-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">模块初始化函数，注册<em>dma</em>_<em>buf</em>_fs，初始化db_list，初始化debugfs。初始化一个attach节点，并把它加入到<em>dma</em><em>buf</em>的attachments列表中。成功会返回&amp;<em>dma</em>_<em>buf</em>的指针，失败会返回一个负数（通过ERR_PTR包装）。创建一个<em>dma</em><em>buf</em>，并把它关联到一个anon <em>file</em>上，以便暴露这块内存。锁住一块<em>dma</em><em>buf</em>。从系统中获取一个可用的<em>fd</em>，并把它跟<em>dma</em><em>buf</em><em>-</em>&gt;<em>file</em>绑定起来。调用用户定义的unmap_<em>dma</em>_<em>buf</em>回调。调用用户定义的map_<em>dma</em>_<em>buf</em>回调。</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/abc3240660/article/details/81942190"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.15","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-15-81942190-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"15","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/abc3240660/article/details/81942190"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/abc3240660/article/details/81942190" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.15","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-15-81942190-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"15","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/abc3240660/article/details/81942190"}'  data-report-query='spm=1001.2101.3001.6650.15&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-15-81942190-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-15-81942190-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">Linux内核笔记之<em>DMA</em>_<em>BUF</em></div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/abc3240660" target="_blank"><span class="blog-title">abc3240660的博客</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">08-22</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					8687
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/abc3240660/article/details/81942190" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.15","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-15-81942190-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"15","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/abc3240660/article/details/81942190"}'  data-report-query='spm=1001.2101.3001.6650.15&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-15-81942190-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-15-81942190-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">
      
        Linux内核笔记之<em>DMA</em>_<em>BUF</em>
        Apr 18, 2018
      
      
      
        
  <em>DMA</em>_<em>BUF</em>    
      需求背景
      概述
      <em>dma</em><em>-</em><em>buf</em>实现
      运作流程
      Importer驱动实例剖析
      Export驱动实例编写
    
  


<em>内存管理</em>...</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://devpress.csdn.net/v1/article/detail/102596772"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.16","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-16-102596772-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"16","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596772"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://devpress.csdn.net/v1/article/detail/102596772" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.16","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-16-102596772-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"16","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596772"}'  data-report-query='spm=1001.2101.3001.6650.16&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-16-102596772-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-16-102596772-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1"><em>dma</em><em>-</em><em>buf</em> <em>由浅入深</em>（三） &mdash;&mdash; map attachment</div>
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
					3万+
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://devpress.csdn.net/v1/article/detail/102596772" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.16","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-16-102596772-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"16","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596772"}'  data-report-query='spm=1001.2101.3001.6650.16&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-16-102596772-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-16-102596772-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">在上一篇《kmap/vmap》中，我们学习了如何使用 CPU 在 kernel 空间访问 <em>dma</em><em>-</em><em>buf</em> 物理内存，但如果使用CPU直接去访问 memory，那么性能会大大降低。因此，<em>dma</em><em>-</em><em>buf</em> 在内核中出现频率最高的还是它的 <em>dma</em>_<em>buf</em>_attach<em>(</em><em>)</em> 和 <em>dma</em>_<em>buf</em>_map_attachment<em>(</em><em>)</em> 接口。本篇我们就一起来学习如何通过这两个 API 来实现 <em>DMA</em> 硬件对 <em>dma</em><em>-</em><em>buf</em> 物理内存的访问。</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://killerp.blog.csdn.net/article/details/139710556"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.17","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-17-139710556-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"17","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://killerp.blog.csdn.net/article/details/139710556"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://killerp.blog.csdn.net/article/details/139710556" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.17","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-17-139710556-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"17","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://killerp.blog.csdn.net/article/details/139710556"}'  data-report-query='spm=1001.2101.3001.6650.17&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-17-139710556-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-17-139710556-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">Linux <em>DMA</em><em>-</em><em>Buf</em>驱动框架</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/weixin_44821644" target="_blank"><span class="blog-title">杀手的博客</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">06-15</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					4146
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://killerp.blog.csdn.net/article/details/139710556" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.17","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-17-139710556-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"17","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://killerp.blog.csdn.net/article/details/139710556"}'  data-report-query='spm=1001.2101.3001.6650.17&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-17-139710556-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-17-139710556-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1"><em>dma</em><em>buf</em> 是一个驱动间共享<em>buf</em> 的机制，他的简单使用场景如下：用户从<em>DRM</em>（显示驱动）申请一个<em>dma</em><em>buf</em>，把<em>dma</em><em>buf</em> 设置给GPU驱动，并启动GPU将数据输出到<em>dma</em><em>buf</em>，GPU输出完成后，再将<em>dma</em><em>buf</em>设置到<em>DRM</em> 驱动，完成画面的显示。在这个过程中通过共享<em>dma</em><em>buf</em>的方式，避免了GPU输出数据拷贝到<em>drm</em> frame <em>buf</em>f的动作。</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/jun_8018/article/details/118895751"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.18","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-18-118895751-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"18","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/jun_8018/article/details/118895751"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/jun_8018/article/details/118895751" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.18","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-18-118895751-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"18","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/jun_8018/article/details/118895751"}'  data-report-query='spm=1001.2101.3001.6650.18&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-18-118895751-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-18-118895751-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1"><em>dma</em><em>-</em><em>buf</em></div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/jun_8018" target="_blank"><span class="blog-title">Jun&#39;s blog</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">07-20</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					597
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/jun_8018/article/details/118895751" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.18","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-18-118895751-blog-102596802.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382492343_98303\"}","dist_request_id":"1766382492343_98303","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"18","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/jun_8018/article/details/118895751"}'  data-report-query='spm=1001.2101.3001.6650.18&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-18-118895751-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-18-118895751-blog-102596802.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">【代码】<em>dma</em><em>-</em><em>buf</em>。</div>
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
            <a href="https://blog.csdn.net/hexiaolong2009/article/details/102596791" data-report-query="spm=3001.10752">
              dma-buf 由浅入深（四） —— mmap
            </a>
          </dd>
      </dl>
      <dl class="next" data-report-click='{"spm":"3001.10796","extend1":"下一篇"}' data-report-view='{"spm":"3001.10796","extend1":"下一篇"}'>
          <dt>
            下一篇：
          </dt>
          <dd >
            <a href="https://blog.csdn.net/hexiaolong2009/article/details/102596825" data-report-query="spm=3001.10796">
              dma-buf 由浅入深（六） —— begin / end cpu_access
            </a>
          </dd>
      </dl>
  </div>
<div id="asideHotArticle" class="aside-box">
	<h3 class="aside-title">大家在看</h3>
	<div class="aside-content">
		<ul class="hotArticle-list">
			<li>
				<a href="https://blog.csdn.net/u014750971/article/details/156139580" target="_blank"  data-report-click='{"spm":"3001.10093","dest":"https://blog.csdn.net/u014750971/article/details/156139580","strategy":"202_1052723-3681435_RCMD","ab":"new"}' data-report-view='{"spm":"3001.10093","dest":"https://blog.csdn.net/u014750971/article/details/156139580","strategy":"202_1052723-3681435_RCMD","ab":"new"}'>
				自监督学习三大范式解析
					<img src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					<span class="read">402</span>
        </a>
			</li>
			<li>
				<a href="https://blog.csdn.net/2509_93766561/article/details/156148540" target="_blank"  data-report-click='{"spm":"3001.10093","dest":"https://blog.csdn.net/2509_93766561/article/details/156148540","strategy":"202_1052723-3681419_RCMD","ab":"new"}' data-report-view='{"spm":"3001.10093","dest":"https://blog.csdn.net/2509_93766561/article/details/156148540","strategy":"202_1052723-3681419_RCMD","ab":"new"}'>
				斗南花市点燃年轻旅游潮
        </a>
			</li>
			<li>
				<a href="https://blog.csdn.net/liteng890116/article/details/156148741" target="_blank"  data-report-click='{"spm":"3001.10093","dest":"https://blog.csdn.net/liteng890116/article/details/156148741","strategy":"202_1052723-3681437_RCMD","ab":"new"}' data-report-view='{"spm":"3001.10093","dest":"https://blog.csdn.net/liteng890116/article/details/156148741","strategy":"202_1052723-3681437_RCMD","ab":"new"}'>
				LLVM从C到ELF的编译流程
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
				<a href="https://blog.csdn.net/weishi122/article/details/156148674" target="_blank"  data-report-click='{"spm":"3001.10093","dest":"https://blog.csdn.net/weishi122/article/details/156148674","strategy":"202_1052723-3681431_RCMD","ab":"new"}' data-report-view='{"spm":"3001.10093","dest":"https://blog.csdn.net/weishi122/article/details/156148674","strategy":"202_1052723-3681431_RCMD","ab":"new"}'>
				WCFM Marketplace授权漏洞致数据泄露
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
            <a href="https://blog.csdn.net/hexiaolong2009/article/details/102596791" data-report-query="spm=3001.10752">
              dma-buf 由浅入深（四） —— mmap
            </a>
          </dd>
      </dl>
      <dl class="next" data-report-click='{"spm":"3001.10796","extend1":"下一篇"}' data-report-view='{"spm":"3001.10796","extend1":"下一篇"}'>
          <dt>
            下一篇：
          </dt>
          <dd >
            <a href="https://blog.csdn.net/hexiaolong2009/article/details/102596825" data-report-query="spm=3001.10796">
              dma-buf 由浅入深（六） —— begin / end cpu_access
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
		<div class="comment-side-tit-count">评论&nbsp;<span class="count">5</span></div>
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
              <input type="hidden" id="article_id" name="article_id" value="102596802">
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
