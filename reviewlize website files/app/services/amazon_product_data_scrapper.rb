class AmazonProductDataScrapper < Kimurai::Base
  def self.scrape(url)
    # self.crawl!
    return self.parse!(:parse, url: url)
  end
  @name = "amazon_product_data_scrapper"
  # @engine = :selenium_chrome
  @start_urls = ["https://www.amazon.com/"]

  def parse(response, url:, data: {})
    image = response.css(".a-dynamic-image-container img").attribute('src').value
    title = response.css("#titleSection").text.squish
    price = response.css("#priceblock_ourprice").text.squish
    url = url.split("?").first.split("ref").first
    
    return [{title: title, price: price, url: url, image: image}]
  end
end